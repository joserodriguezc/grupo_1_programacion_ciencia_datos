import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

# ============================================================
# RUTAS
# ============================================================

# Este archivo está en F2/src/
# parent       -> F2/src
# parent.parent -> F2
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

# Entrada generada por el código 01
VOTACIONES_CSV = DATA_DIR / "interim" / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"

# XML crudos de cada votación
RAW_DIR = DATA_DIR / "raw" / "votaciones"

# CSV consolidado
VOTOS_CSV = DATA_DIR / "interim" / "votaciones" /"detalle_votaciones.csv"


# ============================================================
# API
# ============================================================

BASE_URL = (
    "https://opendata.camara.cl/"
    "camaradiputados/WServices/WSLegislativo.asmx/"
    "retornarVotacionDetalle"
)

NAMESPACE = "http://opendata.camara.cl/camaradiputados/v1"


# ============================================================
# 1. LEER LOS ID OBTENIDOS EN EL CÓDIGO 01
# ============================================================

def obtener_ids_votaciones(
    archivo_csv: Path = VOTACIONES_CSV
) -> list[int]:
    """
    Lee votaciones.csv y obtiene los identificadores únicos
    de las votaciones del proyecto de ley.
    """

    if not archivo_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {archivo_csv}"
        )

    df_votaciones = pd.read_csv(archivo_csv)

    if "Id" not in df_votaciones.columns:
        raise ValueError(
            "El archivo votaciones.csv no contiene la columna 'Id'."
        )

    ids = (
        df_votaciones["Id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    return ids


# ============================================================
# 2. DESCARGAR UN XML POR VOTACIÓN
# ============================================================

def descargar_xml_votaciones(
    ids_votaciones: list[int]
) -> list[Path]:
    """
    Consulta retornarVotacionDetalle para cada identificador
    y conserva cada respuesta XML en data/raw/votaciones.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    archivos_xml = []

    total = len(ids_votaciones)

    for numero, votacion_id in enumerate(
        ids_votaciones,
        start=1
    ):
        print(
            f"[{numero}/{total}] "
            f"Descargando votación {votacion_id}"
        )

        response = requests.get(
            BASE_URL,
            params={
                "prmVotacionId": votacion_id
            },
            timeout=30
        )

        response.raise_for_status()

        # ----------------------------------------------------
        # Detectar HTML de mantenimiento aunque HTTP sea 200
        # ----------------------------------------------------

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        contenido = response.content.lstrip().lower()

        if (
            "html" in content_type
            or contenido.startswith(b"<html")
            or contenido.startswith(b"<!doctype html")
        ):
            raise ValueError(
                f"La votación {votacion_id} devolvió "
                "HTML en lugar de XML."
            )

        # ----------------------------------------------------
        # Comprobar que sea XML válido
        # ----------------------------------------------------

        try:
            ET.fromstring(response.content)

        except ET.ParseError as error:
            raise ValueError(
                f"La votación {votacion_id} "
                "no devolvió un XML válido."
            ) from error

        archivo_xml = (
            RAW_DIR /
            f"votacion_{votacion_id}.xml"
        )

        archivo_xml.write_bytes(
            response.content
        )

        archivos_xml.append(
            archivo_xml
        )

    return archivos_xml


# ============================================================
# FUNCIONES AUXILIARES PARA EL XML
# ============================================================

def _texto(
    elemento: ET.Element | None,
    nombre: str
):
    """
    Obtiene el texto de un elemento hijo considerando
    el namespace de la API.
    """

    if elemento is None:
        return None

    hijo = elemento.find(
        f"{{{NAMESPACE}}}{nombre}"
    )

    if hijo is None:
        return None

    return hijo.text


def _valor_elemento(
    elemento: ET.Element | None
):
    """
    Obtiene el texto de elementos como Quorum,
    Resultado, Tipo u OpcionVoto.
    """

    if elemento is None:
        return None

    return elemento.text


def _codigo_elemento(
    elemento: ET.Element | None
):
    """
    Obtiene el atributo Valor si está disponible.
    """

    if elemento is None:
        return None

    return (
        elemento.get("Valor")
        or elemento.get("valor")
    )


# ============================================================
# 3. LEER UN XML Y CONVERTIRLO EN DATAFRAME
# ============================================================

def leer_xml_votacion(
    archivo_xml: Path
) -> pd.DataFrame:
    """
    Lee un XML de una votación y devuelve un DataFrame
    donde cada fila representa el voto nominal de un diputado.

    También agrega la metadata general de la votación.
    """

    tree = ET.parse(archivo_xml)
    root = tree.getroot()

    namespace = {
        "c": NAMESPACE
    }

    # --------------------------------------------------------
    # Metadata de la votación
    # --------------------------------------------------------

    votacion_id = _texto(
        root,
        "Id"
    )

    descripcion = _texto(
        root,
        "Descripcion"
    )

    fecha = _texto(
        root,
        "Fecha"
    )

    total_si = _texto(
        root,
        "TotalSi"
    )

    total_no = _texto(
        root,
        "TotalNo"
    )

    total_abstencion = _texto(
        root,
        "TotalAbstencion"
    )

    total_dispensado = _texto(
        root,
        "TotalDispensado"
    )

    quorum_elemento = root.find(
        "c:Quorum",
        namespace
    )

    resultado_elemento = root.find(
        "c:Resultado",
        namespace
    )

    tipo_elemento = root.find(
        "c:Tipo",
        namespace
    )

    # --------------------------------------------------------
    # Votos
    # --------------------------------------------------------

    registros = []

    votos = root.findall(
        ".//c:Votos/c:Voto",
        namespace
    )

    for voto in votos:

        diputado = voto.find(
            "c:Diputado",
            namespace
        )

        opcion = voto.find(
            "c:OpcionVoto",
            namespace
        )

        if diputado is None:
            continue

        registro = {
            # Diputado
            "diputado_id": _texto(
                diputado,
                "Id"
            ),
            "nombre": _texto(
                diputado,
                "Nombre"
            ),
            "nombre2": _texto(
                diputado,
                "Nombre2"
            ),
            "apellido_paterno": _texto(
                diputado,
                "ApellidoPaterno"
            ),
            "apellido_materno": _texto(
                diputado,
                "ApellidoMaterno"
            ),

            # Voto
            "opcion_codigo": _codigo_elemento(
                opcion
            ),
            "opcion_voto": _valor_elemento(
                opcion
            ),

            # Metadata de la votación
            "votacion_id": votacion_id,
            "descripcion": descripcion,
            "fecha": fecha,
            "total_si": total_si,
            "total_no": total_no,
            "total_abstencion": total_abstencion,
            "total_dispensado": total_dispensado,

            "quorum_codigo": _codigo_elemento(
                quorum_elemento
            ),
            "quorum": _valor_elemento(
                quorum_elemento
            ),

            "resultado_codigo": _codigo_elemento(
                resultado_elemento
            ),
            "resultado": _valor_elemento(
                resultado_elemento
            ),

            "tipo_codigo": _codigo_elemento(
                tipo_elemento
            ),
            "tipo": _valor_elemento(
                tipo_elemento
            )
        }

        registros.append(
            registro
        )

    return pd.DataFrame(
        registros
    )


# ============================================================
# 4. LEER TODOS LOS XML Y CREAR UN SOLO DATAFRAME
# ============================================================

def consolidar_xml_votaciones(
    archivos_xml: list[Path]
) -> pd.DataFrame:
    """
    Lee todos los XML descargados y concatena los votos
    en un único DataFrame.
    """

    dataframes = []

    for archivo in archivos_xml:
        print(
            f"Leyendo {archivo.name}"
        )

        df = leer_xml_votacion(
            archivo
        )

        if not df.empty:
            dataframes.append(
                df
            )

    if not dataframes:
        return pd.DataFrame()

    df_final = pd.concat(
        dataframes,
        ignore_index=True
    )

    return df_final


# ============================================================
# 5. GUARDAR DATAFRAME CONSOLIDADO
# ============================================================

def guardar_votos_csv(
    df: pd.DataFrame,
    archivo_salida: Path = VOTOS_CSV
) -> None:
    """
    Guarda el DataFrame consolidado en CSV.
    """

    archivo_salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        archivo_salida,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"CSV generado: {archivo_salida}"
    )


# ============================================================
# 6. FUNCIÓN QUE LLAMARÁ EL NOTEBOOK
# ============================================================

def extraer_detalles_votaciones() -> pd.DataFrame:
    """
    Ejecuta el pipeline del paso 04.

    1. Lee los ID desde votaciones.csv.
    2. Descarga un XML por votación.
    3. Guarda los XML en data/raw/votaciones.
    4. Lee los XML.
    5. Construye un único DataFrame.
    6. Genera votos_diputados.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame consolidado con los votos nominales.
    """

    ids_votaciones = obtener_ids_votaciones()

    print(
        f"Votaciones encontradas: "
        f"{len(ids_votaciones)}"
    )

    archivos_xml = descargar_xml_votaciones(
        ids_votaciones
    )

    df_final = consolidar_xml_votaciones(
        archivos_xml
    )

    guardar_votos_csv(
        df_final
    )

    print(
        f"Registros obtenidos: "
        f"{len(df_final)}"
    )

    return df_final

