import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SRC_FILE = Path(__file__).resolve().parents[2] / "src" / "03_diputados.py"


def cargar_modulo():
    spec = importlib.util.spec_from_file_location("diputados", SRC_FILE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def crear_militancia(partido_id, alias):
    return SimpleNamespace(
        Partido=SimpleNamespace(
            Id=partido_id,
            Nombre=f"Partido {alias}",
            Alias=alias,
        ),
        FechaInicio="2022-03-11 00:00:00",
        FechaTermino="2026-03-10 23:59:59",
    )


def crear_diputado_periodo(militancias):
    diputado = SimpleNamespace(
        Id=1001,
        Nombre="Ana",
        Nombre2=None,
        ApellidoPaterno="Pérez",
        ApellidoMaterno="Soto",
        FechaNacimiento="1980-01-01",
        RUT=None,
        RUTDV=None,
        Sexo={"Valor": 2, "_value_1": "Femenino"},
        Militancias=SimpleNamespace(Militancia=militancias),
    )

    return SimpleNamespace(
        Diputado=diputado,
        FechaInicio="2022-03-11 00:00:00",
        FechaTermino="2026-03-10 23:59:59",
    )


def test_construir_filas_diputados_mapea_columnas():
    modulo = cargar_modulo()
    entrada = [crear_diputado_periodo(None)]

    filas = modulo.construir_filas_diputados(entrada, periodo_id=10)

    assert len(filas) == 1
    assert list(filas[0]) == modulo.COLUMNAS_DIPUTADOS
    assert filas[0]["diputado_id"] == 1001
    assert filas[0]["periodo_id"] == 10
    assert filas[0]["sexo_desc"] == "Femenino"


def test_militancias_admite_coleccion_vacia():
    modulo = cargar_modulo()
    filas = modulo.construir_filas_militancias([crear_diputado_periodo(None)])
    assert filas == []


def test_militancias_admite_un_solo_elemento():
    modulo = cargar_modulo()
    militancia = crear_militancia(1, "P1")

    filas = modulo.construir_filas_militancias(
        [crear_diputado_periodo(militancia)]
    )

    assert len(filas) == 1
    assert filas[0]["partido_alias"] == "P1"


def test_militancias_admite_varios_elementos():
    modulo = cargar_modulo()
    militancias = [
        crear_militancia(1, "P1"),
        crear_militancia(2, "P2"),
    ]

    filas = modulo.construir_filas_militancias(
        [crear_diputado_periodo(militancias)]
    )

    assert len(filas) == 2
    assert [fila["partido_alias"] for fila in filas] == ["P1", "P2"]


def test_escribir_csv_respeta_columnas_y_crea_directorio(tmp_path):
    modulo = cargar_modulo()
    filas = modulo.construir_filas_diputados(
        [crear_diputado_periodo(None)],
        periodo_id=10,
    )
    ruta = tmp_path / "salida" / "diputados.csv"

    modulo.escribir_csv(filas, modulo.COLUMNAS_DIPUTADOS, ruta)

    assert ruta.exists()
    guardado = pd.read_csv(ruta)
    assert list(guardado.columns) == modulo.COLUMNAS_DIPUTADOS
    assert len(guardado) == 1
