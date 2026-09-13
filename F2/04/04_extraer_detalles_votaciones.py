from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_URL = (
    "https://opendata.camara.cl/"
    "camaradiputados/WServices/WSLegislativo.asmx/"
    "retornarVotacionDetalle"
)

DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "interim" / "votaciones.csv"

RAW_DIR = DATA_DIR / "raw" / "detalle_votaciones"

DETALLE_FILE = DATA_DIR / "interim" / "detalle_votaciones.csv"
VOTOS_FILE = DATA_DIR / "interim" / "votos_diputados.csv"

NAMESPACE = {
    "camara": "http://opendata.camara.cl/camaradiputados/v1"
}


# ============================================================
# DESCARGA DEL XML
# ============================================================

def descargar_xml_votacion(votacion_id: int) -> Path:
    """
    Consulta retornarVotacionDetalle para una votación
    y almacena la respuesta XML en data/raw/detalle_votaciones.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "prmVotacionId": votacion_id
    }

    print(f"Descargando votación {votacion_id}...")

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # Validar que no sea una página HTML de mantenimiento
    # --------------------------------------------------------

    content_type = response.headers.get("Content-Type", "").lower()
    contenido = response.content.lstrip().lower()

    if "html" in content_type:
        raise ValueError(
            f"La votación {votacion_id} devolvió HTML en lugar de XML."
        )

    if (
        contenido.startswith(b"<html")
        or contenido.startswith(b"<!doctype html")
    ):
        raise ValueError(
            f"La votación {votacion_id} devolvió una página HTML."
        )

    # --------------------------------------------------------
    # Validar que el contenido sea XML válido
    # --------------------------------------------------------

    try:
        ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise ValueError(
            f"La respuesta de la votación {votacion_id} no es XML válido."
        ) from exc

    xml_file = RAW_DIR / f"votacion_{votacion_id}.xml"

    xml_file.write_bytes(response.content)

    return xml_file


# ============================================================
# EXTRAER DATOS GENERALES DE LA VOTACIÓN
# ============================================================

def extraer_detalle_votacion(xml_file: Path) -> pd.DataFrame:
    """
    Extrae los datos generales de una votación.
    """

    df = pd.read_xml(
        xml_file,
        xpath="/camara:Votacion",
        namespaces=NAMESPACE
    )

    columnas = [
        "Id",
        "Descripcion",
        "Fecha",
        "TotalSi",
        "TotalNo",
        "TotalAbstencion",
        "TotalDispensado",
        "Quorum",
        "Resultado",
        "Tipo"
    ]

    columnas_existentes = [
        columna
        for columna in columnas
        if columna in df.columns
    ]

    return df[columnas_existentes]


# ============================================================
# EXTRAER VOTOS DE LOS DIPUTADOS
# ============================================================

def extraer_votos_diputados(
    xml_file: Path,
    votacion_id: int
) -> pd.DataFrame:
    """
    Extrae el voto nominal de cada diputado para una votación.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    namespace = (
        "{http://opendata.camara.cl/"
        "camaradiputados/v1}"
    )

    registros = []

    votos = root.findall(
        f".//{namespace}Votos/{namespace}Voto"
    )

    for voto in votos:

        diputado = voto.find(f"{namespace}Diputado")
        opcion_voto = voto.find(f"{namespace}OpcionVoto")

        if diputado is None:
            continue

        def obtener_texto(nombre):
            elemento = diputado.find(f"{namespace}{nombre}")

            if elemento is None:
                return None

            return elemento.text

        registros.append(
            {
                "votacion_id": votacion_id,
                "diputado_id": obtener_texto("Id"),
                "nombre": obtener_texto("Nombre"),
                "nombre2": obtener_texto("Nombre2"),
                "apellido_paterno": obtener_texto(
                    "ApellidoPaterno"
                ),
                "apellido_materno": obtener_texto(
                    "ApellidoMaterno"
                ),
                "rut": obtener_texto("RUT"),
                "rut_dv": obtener_texto("RUTDV"),
                "opcion_voto": (
                    opcion_voto.text
                    if opcion_voto is not None
                    else None
                )
            }
        )

    return pd.DataFrame(registros)


# ============================================================
# PIPELINE
# ============================================================

def extraer_detalles_votaciones():
    """
    Lee votaciones.csv, consulta el detalle de cada votación
    y genera los CSV de detalle y votos nominales.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {INPUT_FILE}"
        )

    votaciones = pd.read_csv(INPUT_FILE)

    if "Id" not in votaciones.columns:
        raise ValueError(
            "votaciones.csv no contiene la columna 'Id'."
        )

    ids_votaciones = (
        votaciones["Id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Se encontraron {len(ids_votaciones)} votaciones."
    )

    detalles = []
    votos_diputados = []

    for numero, votacion_id in enumerate(
        ids_votaciones,
        start=1
    ):
        print(
            f"[{numero}/{len(ids_votaciones)}] "
            f"Procesando votación {votacion_id}"
        )

        try:
            # 1. Descargar y guardar XML raw
            xml_file = descargar_xml_votacion(
                votacion_id
            )

            # 2. Extraer información general
            df_detalle = extraer_detalle_votacion(
                xml_file
            )

            detalles.append(df_detalle)

            # 3. Extraer votos individuales
            df_votos = extraer_votos_diputados(
                xml_file,
                votacion_id
            )

            votos_diputados.append(df_votos)

        except Exception as exc:
            print(
                f"ERROR en votación {votacion_id}: "
                f"{exc}"
            )

    # ========================================================
    # CONSOLIDAR
    # ========================================================

    if detalles:
        df_detalles = pd.concat(
            detalles,
            ignore_index=True
        )
    else:
        df_detalles = pd.DataFrame()

    if votos_diputados:
        df_votos = pd.concat(
            votos_diputados,
            ignore_index=True
        )
    else:
        df_votos = pd.DataFrame()

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    DETALLE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_detalles.to_csv(
        DETALLE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    df_votos.to_csv(
        VOTOS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nProceso terminado.")
    print(f"Detalle votaciones: {DETALLE_FILE}")
    print(f"Votos diputados:    {VOTOS_FILE}")

    print(
        f"Votaciones procesadas: "
        f"{len(df_detalles)}"
    )

    print(
        f"Votos nominales obtenidos: "
        f"{len(df_votos)}"
    )

    return df_detalles, df_votos


# ============================================================
# MAIN
# ============================================================

def main():
    extraer_detalles_votaciones()


if __name__ == "__main__":
    main()