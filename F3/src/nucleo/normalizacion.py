"""Conversiones explícitas entre los CSV interim y las tablas procesadas."""

import pandas as pd


def normalizar_identificador(serie: pd.Series, columna: str) -> pd.Series:
    resultado = serie.astype("string").str.strip()
    invalidos = resultado.isna() | resultado.eq("").fillna(False)
    if invalidos.any():
        raise ValueError(
            f"{columna}: identificador vacío en filas {serie.index[invalidos].tolist()}"
        )
    return resultado


def normalizar_fecha(serie: pd.Series, columna: str, *, permitir_nulos: bool = False) -> pd.Series:
    """Rechaza fechas ilegibles y, salvo autorización expresa, fechas ausentes."""
    vacios = serie.isna() | serie.astype("string").str.strip().eq("").fillna(False)
    resultado = pd.to_datetime(serie, errors="coerce", format="mixed")
    invalidos = resultado.isna() & (~vacios if permitir_nulos else True)
    if invalidos.any():
        raise ValueError(f"{columna}: fecha inválida en filas {serie.index[invalidos].tolist()}")
    return resultado


def normalizar_total(serie: pd.Series, columna: str) -> pd.Series:
    resultado = pd.to_numeric(serie, errors="coerce")
    invalidos = resultado.isna() | (resultado < 0) | resultado.mod(1).ne(0)
    if invalidos.any():
        raise ValueError(f"{columna}: total inválido en filas {serie.index[invalidos].tolist()}")
    return resultado.astype("Int64")
