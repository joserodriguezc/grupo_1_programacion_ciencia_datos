"""Congelamiento del corte F3: resumen, validación, manifiesto y errores."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from F4.src.analisis import corte
from F4.src.analisis.corte import (
    ErrorCorteF4,
    ErrorValidacionCorte,
    ResumenCorte,
    calcular_sha256,
    cargar_entrada,
    construir_manifiesto,
    guardar_manifiesto,
    resumir_corte,
    validar_corte,
)

RAIZ = Path(__file__).resolve().parents[3]
CSV = "diputado_id,votacion_id,opcion\n1,10,Si\n2,10,No\n1,11,Si\n"


@pytest.fixture
def entrada(tmp_path: Path) -> Path:
    ruta = tmp_path / "big_table.csv"
    ruta.write_bytes(CSV.encode("utf-8"))
    return ruta


def _resumen(filas=3, columnas=3, votaciones=2) -> ResumenCorte:
    return ResumenCorte("x.csv", "0" * 64, filas, columnas, votaciones, "votacion_id", {})


def test_sha256_coincide_con_hashlib_y_no_depende_del_bloque(entrada: Path) -> None:
    esperado = hashlib.sha256(CSV.encode("utf-8")).hexdigest()
    assert calcular_sha256(entrada) == esperado
    assert calcular_sha256(entrada, tamano_bloque=4) == esperado


def test_cargar_entrada_valida_la_ruta(entrada: Path, tmp_path: Path) -> None:
    assert cargar_entrada(entrada).shape == (3, 3)
    with pytest.raises(FileNotFoundError):
        cargar_entrada(tmp_path / "no_existe.csv")
    with pytest.raises(ErrorCorteF4):
        cargar_entrada(tmp_path)  # un directorio no es una entrada


def test_resumir_corte_registra_dimensiones_esquema_y_hash(entrada: Path) -> None:
    df = pd.read_csv(entrada)
    resumen = resumir_corte(df, entrada, Path("F3/data/processed/big_table.csv"))

    assert (resumen.filas, resumen.columnas, resumen.votaciones) == (3, 3, 2)
    assert resumen.ruta == "F3/data/processed/big_table.csv"
    assert resumen.sha256 == calcular_sha256(entrada)
    assert resumen.esquema == {"diputado_id": "int64", "votacion_id": "int64",
                               "opcion": str(df["opcion"].dtype)}


def test_resumir_corte_exige_la_columna_de_votacion(entrada: Path) -> None:
    df = pd.read_csv(entrada).drop(columns="votacion_id")
    with pytest.raises(ErrorValidacionCorte, match="votacion_id"):
        resumir_corte(df, entrada, entrada)


def test_validar_corte_lista_cada_discrepancia() -> None:
    assert validar_corte(_resumen(), filas_esperadas=3, columnas_esperadas=3,
                         votaciones_esperadas=2) == []
    discrepancias = validar_corte(_resumen(filas=4, votaciones=1), filas_esperadas=3,
                                  columnas_esperadas=3, votaciones_esperadas=2)
    assert discrepancias == ["filas: observadas=4, esperadas=3",
                             "votaciones: observadas=1, esperadas=2"]


def test_variacion_sin_justificacion_bloquea_el_manifiesto(tmp_path: Path) -> None:
    with pytest.raises(ErrorValidacionCorte, match="justificación"):
        construir_manifiesto(raiz=tmp_path, resumen=_resumen(), discrepancias=["filas: x"])


def test_manifiesto_con_variacion_justificada(monkeypatch) -> None:
    monkeypatch.setattr(corte, "obtener_estado_git",
                        lambda raiz: {"rama": "main", "working_tree_limpio": True})
    monkeypatch.setattr(corte, "obtener_commit_git", lambda raiz: "abc123")

    manifiesto = construir_manifiesto(raiz=RAIZ, resumen=_resumen(),
                                      discrepancias=["filas: x"],
                                      justificacion_variacion="Corte ampliado por F3.")

    assert manifiesto["validacion"]["estado"] == "variacion_justificada"
    assert manifiesto["reproducibilidad"]["commit_git"] == "abc123"
    assert manifiesto["entrada"]["sha256"] == "0" * 64
    assert manifiesto["reglas"]["transformaciones_aplicadas"] is False


def test_fuera_de_un_repositorio_git_falla_con_error_propio(tmp_path: Path) -> None:
    with pytest.raises(ErrorCorteF4):
        corte.obtener_commit_git(tmp_path)
    with pytest.raises(ErrorCorteF4):
        corte.obtener_estado_git(tmp_path)


def test_guardar_manifiesto_es_determinista(tmp_path: Path) -> None:
    ruta = tmp_path / "reportes" / "manifiesto.json"
    guardar_manifiesto({"b": 1, "a": "ñ"}, ruta)
    texto = ruta.read_text(encoding="utf-8")

    assert texto == '{\n  "a": "ñ",\n  "b": 1\n}\n'
    assert json.loads(texto) == {"a": "ñ", "b": 1}


def test_corte_versionado_coincide_con_su_manifiesto() -> None:
    manifiesto = json.loads(
        (RAIZ / "F4/data/reports/manifiesto_corte.json").read_text(encoding="utf-8"))
    entrada = RAIZ / manifiesto["entrada"]["ruta"]
    resumen = resumir_corte(cargar_entrada(entrada), entrada, Path(manifiesto["entrada"]["ruta"]))

    assert resumen.sha256 == manifiesto["entrada"]["sha256"]
    assert validar_corte(resumen) == manifiesto["validacion"]["discrepancias"] == []
