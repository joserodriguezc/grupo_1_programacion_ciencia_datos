import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd

SRC_FILE = Path(__file__).resolve().parents[2] / "src" / "02_periodos_legislativos.py"

COLUMNAS_ESPERADAS = [
    "periodo_id",
    "nombre",
    "fecha_inicio",
    "fecha_termino",
]


def cargar_modulo():
    spec = importlib.util.spec_from_file_location("periodos_legislativos", SRC_FILE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def periodos_ejemplo():
    return [
        {
            "Id": 10,
            "Nombre": "2022-2026",
            "FechaInicio": datetime(2022, 3, 11),
            "FechaTermino": datetime(2026, 3, 10, 23, 59, 59),
        },
        {
            "Id": 11,
            "Nombre": "2026-2030",
            "FechaInicio": datetime(2026, 3, 11),
            "FechaTermino": datetime(2030, 3, 10, 23, 59, 59),
        },
    ]


def test_construir_filas_periodos_mapea_campos_y_fechas_limite():
    modulo = cargar_modulo()
    filas = modulo.construir_filas_periodos(periodos_ejemplo())

    assert len(filas) == 2
    assert list(filas[0]) == COLUMNAS_ESPERADAS
    assert filas[0]["periodo_id"] == 10
    assert filas[0]["fecha_inicio"] == datetime(2022, 3, 11)
    assert filas[0]["fecha_termino"] == datetime(2026, 3, 10, 23, 59, 59)


def test_extraer_periodos_escribe_csv_sin_red(tmp_path, monkeypatch):
    modulo = cargar_modulo()

    class ServicioFalso:
        @staticmethod
        def retornarPeriodosLegislativos():
            return list(reversed(periodos_ejemplo()))

    class ClienteFalso:
        def __init__(self, _wsdl):
            self.service = ServicioFalso()

    monkeypatch.setattr(modulo, "Client", ClienteFalso)
    monkeypatch.setattr(modulo, "INTERIM_DIR", tmp_path)

    resultado = modulo.extraer_periodos()
    ruta = tmp_path / "periodos.csv"

    assert ruta.exists()
    assert list(resultado.columns) == COLUMNAS_ESPERADAS
    assert resultado["periodo_id"].tolist() == [10, 11]

    guardado = pd.read_csv(ruta)
    assert list(guardado.columns) == COLUMNAS_ESPERADAS
    assert len(guardado) == 2
