"""
Extrae todos los períodos legislativos desde retornarPeriodosLegislativos
(API SOAP de la Cámara de Diputados) y los guarda en data/interim/periodos.csv.

No guarda XML crudo: el volumen es bajo (11 períodos) y no requiere
conservar evidencia binaria como en diputados o votaciones.

Este módulo no se ejecuta solo: expone extraer_periodos() para ser
llamada desde el notebook orquestador (F2_01_Obtencion.ipynb).
"""

from pathlib import Path

import pandas as pd
from zeep import Client

WSDL_LEGISLATIVO = "https://opendata.camara.cl/camaradiputados/WServices/WSLegislativo.asmx?WSDL"

# Este archivo está en F2/src/
# parent        -> F2/src
# parent.parent -> F2
BASE_DIR = Path(__file__).resolve().parent.parent
INTERIM_DIR = BASE_DIR / "data" / "interim"


def construir_filas_periodos(periodos):
    """
    A partir de la lista de Periodo que devuelve zeep, arma una
    fila por período legislativo.
    """
    return [
        {
            "periodo_id": p["Id"],
            "nombre": p["Nombre"],
            "fecha_inicio": p["FechaInicio"],
            "fecha_termino": p["FechaTermino"],
        }
        for p in periodos
    ]


def extraer_periodos():
    """
    Ejecuta el flujo completo: consulta la API, construye el
    DataFrame de períodos ordenado por fecha de inicio, y lo
    guarda como periodos.csv. No retorna nada; el resultado
    queda en disco.
    """
    client_legislativo = Client(WSDL_LEGISLATIVO)
    periodos = client_legislativo.service.retornarPeriodosLegislativos()

    filas = construir_filas_periodos(periodos)
    df_periodos = pd.DataFrame(filas)
    df_periodos = df_periodos.sort_values("fecha_inicio").reset_index(drop=True)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    ruta_csv = INTERIM_DIR / "periodos.csv"
    df_periodos.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"Guardado en: {ruta_csv.resolve()} ({len(df_periodos)} filas)")

    return df_periodos