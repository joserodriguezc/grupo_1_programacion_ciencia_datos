import csv
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from F3.src.nucleo.modelos import (
    Diputado,
    Militancia,
    PeriodoLegislativo,
    VotacionProyecto,
    VotoNominal,
)

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]

#ESTAS RUTAS HAY QUE CAMBIARLAS PARA QUE APUNTEN A LA CARPETA F3/data/interim
@pytest.mark.parametrize(
    ("modelo", "ruta_relativa"),
    [
        (PeriodoLegislativo, "F2/data/interim/periodos.csv"),
        (Diputado, "F2/data/interim/diputados.csv"),
        (Militancia, "F2/data/interim/militancias.csv"),
        (VotacionProyecto, "F2/data/interim/VotacionesPorProyectoDeLey/proyecto_ley.csv"),
        (VotoNominal,"F2/data/interim/votaciones/detalle_votaciones.csv")
        
    ],
)
def test_filas_f2_conservan_sus_valores_al_generar_csv(
    modelo: type,
    ruta_relativa: str,
) -> None:
    ruta = RAIZ_REPOSITORIO / ruta_relativa

    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        filas_f2 = list(csv.DictReader(archivo))

    filas_validadas = [
        modelo.from_dict(fila).to_interim_dict()
        for fila in filas_f2
    ]

    salida = StringIO()
    pd.DataFrame(
        filas_validadas,
        columns=modelo.COLUMNAS,
    ).to_csv(salida, index=False)

    salida.seek(0)
    lector = csv.DictReader(salida)

    assert lector.fieldnames == list(modelo.COLUMNAS)
    assert list(lector) == filas_f2


def test_periodo_rechaza_fecha_invalida() -> None:
    fila = {
        "periodo_id": 7,
        "nombre": "1965-1969",
        "fecha_inicio": "1965-03-11",
        "fecha_termino": "1969-13-11",
    }

    with pytest.raises(ValueError, match="fecha_termino: fecha ISO inválida"):
        PeriodoLegislativo.from_dict(fila)


def test_periodo_rechaza_columna_faltante() -> None:
    fila = {
        "periodo_id": 7,
        "nombre": "1965-1969",
        "fecha_inicio": "1965-03-11",
    }

    with pytest.raises(ValueError, match="fecha_termino"):
        PeriodoLegislativo.from_dict(fila)


def test_diputado_convierte_nulos_permitidos_a_campos_vacios() -> None:
    fila = {
        "diputado_id": 1096,
        "nombre": "María Candelaria",
        "nombre2": None,
        "apellido_paterno": "Acevedo",
        "apellido_materno": "Sáez",
        "fecha_nacimiento": "1958-09-12 00:00:00",
        "rut": None,
        "rut_dv": None,
        "sexo_valor": 0,
        "sexo_desc": "Femenino",
        "periodo_id": 10,
        "fecha_inicio_periodo": "2026-03-10 23:59:59",
        "fecha_termino_periodo": None,
    }

    diputado = Diputado.from_dict(fila)
    salida = diputado.to_interim_dict()

    assert salida["diputado_id"] == "1096"
    assert salida["nombre2"] == ""
    assert salida["rut"] == ""
    assert salida["rut_dv"] == ""
    assert salida["fecha_termino_periodo"] == ""
    assert salida["fecha_inicio_periodo"] == "2026-03-10 23:59:59"


def test_diputado_rechaza_identificador_invalido() -> None:
    fila = {
        "diputado_id": "desconocido",
        "nombre": "María Candelaria",
        "nombre2": "",
        "apellido_paterno": "Acevedo",
        "apellido_materno": "Sáez",
        "fecha_nacimiento": "1958-09-12 00:00:00",
        "rut": "",
        "rut_dv": "",
        "sexo_valor": "0",
        "sexo_desc": "Femenino",
        "periodo_id": "10",
        "fecha_inicio_periodo": "2026-03-10 23:59:59",
        "fecha_termino_periodo": "",
    }

    with pytest.raises(ValueError, match="diputado_id"):
        Diputado.from_dict(fila)


def test_diputado_es_inmutable() -> None:
    fila = {
        "diputado_id": "1096",
        "nombre": "María Candelaria",
        "nombre2": "",
        "apellido_paterno": "Acevedo",
        "apellido_materno": "Sáez",
        "fecha_nacimiento": "1958-09-12 00:00:00",
        "rut": "",
        "rut_dv": "",
        "sexo_valor": "0",
        "sexo_desc": "Femenino",
        "periodo_id": "10",
        "fecha_inicio_periodo": "2026-03-10 23:59:59",
        "fecha_termino_periodo": "",
    }

    diputado = Diputado.from_dict(fila)

    with pytest.raises(FrozenInstanceError):
        setattr(diputado, "nombre", "Otro nombre")

def test_militancia_acepta_codigo_textual_y_vigencia_abierta() -> None:
    fila = {
        "diputado_id": 1096,
        "partido_id": "PC",
        "partido_nombre": "Partido Comunista",
        "partido_alias": "PC",
        "fecha_inicio": "2026-03-11 00:00:00",
        "fecha_termino": None,
    }

    salida = Militancia.from_dict(fila).to_interim_dict()

    assert salida["diputado_id"] == "1096"
    assert salida["partido_id"] == "PC"
    assert salida["fecha_termino"] == ""


def test_militancia_rechaza_fecha_inicio_invalida() -> None:
    fila = {
        "diputado_id": "1096",
        "partido_id": "PC",
        "partido_nombre": "Partido Comunista",
        "partido_alias": "PC",
        "fecha_inicio": "2026-15-11",
        "fecha_termino": "2030-03-10 23:59:59",
    }

    with pytest.raises(ValueError, match="fecha_inicio: fecha ISO inválida"):
        Militancia.from_dict(fila)
        
def test_votacion_proyecto_acepta_articulo_vacio() -> None:
    ruta = (
        RAIZ_REPOSITORIO
        / "F2/data/interim/VotacionesPorProyectoDeLey/proyecto_ley.csv"
    )
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        fila = next(csv.DictReader(archivo))

    fila["Articulo"] = None
    salida = VotacionProyecto.from_dict(fila).to_interim_dict()

    assert salida["Articulo"] == ""
    assert salida["Fecha"] == "2024-08-26T19:03:25"


def test_votacion_proyecto_rechaza_conteo_negativo() -> None:
    ruta = (
        RAIZ_REPOSITORIO
        / "F2/data/interim/VotacionesPorProyectoDeLey/proyecto_ley.csv"
    )
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        fila = next(csv.DictReader(archivo))

    fila["TotalSi"] = -1

    with pytest.raises(ValueError, match="TotalSi"):
        VotacionProyecto.from_dict(fila)

def test_voto_nominal_acepta_nombre2_vacio() -> None:
    ruta = (
        RAIZ_REPOSITORIO
        / "F2/data/interim/votaciones/detalle_votaciones.csv"
    )
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        fila = next(csv.DictReader(archivo))

    fila["nombre2"] = None
    salida = VotoNominal.from_dict(fila).to_interim_dict()

    assert salida["nombre2"] == ""
    assert salida["fecha"] == "2024-08-26T19:03:25"


def test_voto_nominal_rechaza_total_negativo() -> None:
    ruta = (
        RAIZ_REPOSITORIO
        / "F2/data/interim/votaciones/detalle_votaciones.csv"
    )
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        fila = next(csv.DictReader(archivo))

    fila["total_no"] = -1

    with pytest.raises(ValueError, match="total_no"):
        VotoNominal.from_dict(fila)