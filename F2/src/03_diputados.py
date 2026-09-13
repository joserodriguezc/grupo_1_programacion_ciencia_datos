"""
Extrae diputados y militancias por período legislativo desde
retornarDiputadosXPeriodo (API SOAP de la Cámara de Diputados).

Guarda el XML crudo (envelope SOAP) en data/raw/diputados/ y
construye diputados.csv y militancias.csv en data/interim/.

Este módulo no se ejecuta solo: expone extraer_diputados(periodo_id)
para ser llamada desde el notebook orquestador (F2_01_Obtencion.ipynb),
que entrega el período como parámetro.
"""

import csv
from pathlib import Path
from zeep import Client
from zeep.plugins import HistoryPlugin
from lxml import etree

WSDL_DIPUTADO = "https://opendata.camara.cl/camaradiputados/WServices/WSDiputado.asmx?WSDL"

# Este archivo está en F2/src/
# parent        -> F2/src
# parent.parent -> F2
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "diputados"
INTERIM_DIR = BASE_DIR / "data" / "interim"

COLUMNAS_DIPUTADOS = [
    "diputado_id", "nombre", "nombre2", "apellido_paterno", "apellido_materno",
    "fecha_nacimiento", "rut", "rut_dv", "sexo",
    "periodo_id", "fecha_inicio_periodo", "fecha_termino_periodo",
]
COLUMNAS_MILITANCIAS = [
    "diputado_id", "partido_id", "partido_nombre", "partido_alias",
    "fecha_inicio", "fecha_termino",
]


def construir_filas_diputados(diputados_periodo, periodo_id):
    """
    A partir de la lista de DiputadoPeriodo que devuelve zeep,
    arma una fila por diputado con sus datos personales y su
    vigencia dentro del período consultado.
    """
    filas = []
    for dp in diputados_periodo:
        d = dp.Diputado
        filas.append({
            "diputado_id": d.Id,
            "nombre": d.Nombre,
            "nombre2": d.Nombre2,
            "apellido_paterno": d.ApellidoPaterno,
            "apellido_materno": d.ApellidoMaterno,
            "fecha_nacimiento": d.FechaNacimiento,
            "rut": d.RUT,
            "rut_dv": d.RUTDV,
            "sexo": d.Sexo,
            "periodo_id": periodo_id,
            "fecha_inicio_periodo": dp.FechaInicio,
            "fecha_termino_periodo": dp.FechaTermino,
        })
    return filas


def construir_filas_militancias(diputados_periodo):
    """
    A partir de la misma lista, arma una fila por cada militancia
    de cada diputado (historial completo, sin filtrar por período).
    """
    filas = []
    for dp in diputados_periodo:
        d = dp.Diputado
        for m in d.Militancias.Militancia:
            filas.append({
                "diputado_id": d.Id,
                "partido_id": m.Partido.Id,
                "partido_nombre": m.Partido.Nombre,
                "partido_alias": m.Partido.Alias,
                "fecha_inicio": m.FechaInicio,
                "fecha_termino": m.FechaTermino,
            })
    return filas


def escribir_csv(filas, columnas, ruta_archivo):
    """Escribe una lista de dicts a CSV, creando la carpeta si falta."""
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    with ruta_archivo.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)
    print(f"Guardado en: {ruta_archivo.resolve()} ({len(filas)} filas)")


def extraer_diputados(periodo_id):
    """
    Ejecuta el flujo completo para un período legislativo dado:
    consulta la API, guarda el XML crudo y genera diputados.csv
    y militancias.csv. No retorna nada; el resultado queda en disco.
    """
    # Conexión y llamada al servicio web
    history = HistoryPlugin()
    client_diputado = Client(WSDL_DIPUTADO, plugins=[history])

    resultado = client_diputado.service.retornarDiputadosXPeriodo(prmPeriodoID=str(periodo_id))

    # Extraer y guardar el envelope SOAP crudo
    ultimo = history.last_received
    envelope = ultimo["envelope"]

    xml_crudo = etree.tostring(
        envelope, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    ).decode("utf-8")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ruta_xml = RAW_DIR / f"diputados_periodo_{periodo_id}.xml"
    ruta_xml.write_text(xml_crudo, encoding="utf-8")
    print(f"Guardado en: {ruta_xml.resolve()}")

    # Construcción y escritura de los CSV a partir del objeto ya parseado por zeep
    filas_diputados = construir_filas_diputados(resultado, periodo_id)
    filas_militancias = construir_filas_militancias(resultado)

    escribir_csv(filas_diputados, COLUMNAS_DIPUTADOS, INTERIM_DIR / "diputados.csv")
    escribir_csv(filas_militancias, COLUMNAS_MILITANCIAS, INTERIM_DIR / "militancias.csv")