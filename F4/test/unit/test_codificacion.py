from pathlib import Path

import pandas as pd
import pytest

from F4.src.analisis.codificacion import (
    CodificadorVotos,
    ErrorCodificacion,
    cargar_diccionario,
)


RAIZ = Path(__file__).resolve().parents[3]
DICCIONARIO = RAIZ / "F4/config/diccionario_votos.toml"


@pytest.fixture
def codificador() -> CodificadorVotos:
    return CodificadorVotos(cargar_diccionario(DICCIONARIO))


def _tabla_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diputado_id": [10, 11, 12],
            "votacion_id": [100, 100, 100],
            "opcion_codigo": [1, 0, 2],
            "opcion_voto": ["Afirmativo", "En Contra", "Abstención"],
            "otra_columna": ["a", "b", "c"],
        }
    )


def test_construye_tres_vistas_sin_perder_columnas(codificador: CodificadorVotos) -> None:
    entrada = _tabla_base()
    salida = codificador.transformar(entrada)

    assert list(salida["voto_binario"].astype("object")) == [1, 0, pd.NA]
    assert salida["voto_ternario"].tolist() == [1, -1, 0]
    assert salida["voto_nominal"].tolist() == ["Sí", "No", "Abstención"]
    assert salida["observado"].tolist() == [True, True, True]
    assert salida["estado_observacion"].tolist() == ["observado"] * 3
    assert salida["otra_columna"].tolist() == entrada["otra_columna"].tolist()
    assert entrada.columns.tolist() == [
        "diputado_id",
        "votacion_id",
        "opcion_codigo",
        "opcion_voto",
        "otra_columna",
    ]


def test_na_no_se_convierte_en_abstencion(codificador: CodificadorVotos) -> None:
    tabla = pd.DataFrame(
        {
            "diputado_id": [10],
            "votacion_id": [100],
            "opcion_codigo": [pd.NA],
            "opcion_voto": [pd.NA],
        }
    )

    salida = codificador.transformar(tabla)
    fila = salida.iloc[0]

    assert pd.isna(fila["voto_binario"])
    assert pd.isna(fila["voto_ternario"])
    assert pd.isna(fila["voto_nominal"])
    assert bool(fila["observado"]) is False
    assert fila["estado_observacion"] == "desconocido"


def test_abstencion_es_observada_pero_no_binaria(codificador: CodificadorVotos) -> None:
    salida = codificador.transformar(_tabla_base())
    abstencion = salida.loc[salida["opcion_codigo"].eq(2)].iloc[0]

    assert bool(abstencion["observado"]) is True
    assert abstencion["voto_ternario"] == 0
    assert pd.isna(abstencion["voto_binario"])
    assert abstencion["voto_nominal"] == "Abstención"


def test_rechaza_codigo_no_declarado(codificador: CodificadorVotos) -> None:
    tabla = pd.DataFrame(
        {
            "diputado_id": [10],
            "votacion_id": [100],
            "opcion_codigo": [9],
            "opcion_voto": ["Otro"],
        }
    )

    with pytest.raises(ErrorCodificacion, match="no declarados"):
        codificador.transformar(tabla)


def test_rechaza_cambio_de_significado_del_codigo(codificador: CodificadorVotos) -> None:
    tabla = pd.DataFrame(
        {
            "diputado_id": [10],
            "votacion_id": [100],
            "opcion_codigo": [1],
            "opcion_voto": ["En Contra"],
        }
    )

    with pytest.raises(ErrorCodificacion, match="no coincide"):
        codificador.transformar(tabla)


def test_rechaza_clave_duplicada(codificador: CodificadorVotos) -> None:
    tabla = pd.concat([_tabla_base().iloc[[0]], _tabla_base().iloc[[0]]], ignore_index=True)

    with pytest.raises(ErrorCodificacion, match="debe ser única"):
        codificador.transformar(tabla)


def test_cubre_todos_los_valores_reales_del_corte(codificador: CodificadorVotos) -> None:
    entrada = pd.read_csv(RAIZ / "F3/data/processed/big_table_analitica.csv")
    salida = codificador.transformar(entrada)

    assert len(salida) == 1993
    assert salida["observado"].all()
    assert set(salida["voto_nominal"].dropna().unique()) == {"Sí", "No", "Abstención"}
    assert salida.loc[salida["voto_nominal"].eq("Abstención"), "voto_binario"].isna().all()
