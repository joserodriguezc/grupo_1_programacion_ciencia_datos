from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def buscar_duplicados_clave(
    df: pd.DataFrame,
    columnas: Iterable[str],
) -> pd.DataFrame:
    """Retorna las filas que repiten una clave simple o compuesta."""
    columnas = list(columnas)
    faltantes = [columna for columna in columnas if columna not in df.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas para validar la clave: {faltantes}")

    mascara = df.duplicated(subset=columnas, keep=False)
    return df.loc[mascara].copy()

class IntervaloInvertidoError(ValueError):
    """La fecha de inicio es posterior a la de término."""

    def __init__(self, columna: str, filas: tuple[object, ...]) -> None:
        self.filas = filas
        super().__init__(
            f"{columna}: intervalo invertido en filas {list(filas[:10])}"
        )

def detectar_solapamientos_vigencia(
    df: pd.DataFrame,
    *,
    id_col: str = "diputado_id",
    inicio_col: str = "fecha_inicio",
    termino_col: str = "fecha_termino",
) -> pd.DataFrame:
    """Detecta solapamientos con extremos incluidos: [inicio, término].

    Un término vacío representa vigencia abierta; una fecha inválida es un error.
    """
    requeridas = {id_col, inicio_col, termino_col}
    faltantes = requeridas - set(df.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas temporales: {sorted(faltantes)}")

    trabajo = df.copy()
    inicio_original = trabajo[inicio_col]
    termino_original = trabajo[termino_col]

    termino_abierto = termino_original.isna() | (
        termino_original.astype("string").str.strip().eq("").fillna(False)
    )

    trabajo["_inicio"] = pd.to_datetime(
        inicio_original, errors="coerce", format="mixed"
    )
    trabajo["_termino"] = pd.to_datetime(
        termino_original, errors="coerce", format="mixed"
    )

    for columna, mascara in (
        (inicio_col, trabajo["_inicio"].isna()),
        (termino_col, trabajo["_termino"].isna() & ~termino_abierto),
    ):
        if mascara.any():
            indices = trabajo.index[mascara].tolist()
            raise ValueError(
                f"{columna}: fecha inválida en filas {indices[:10]}"
            )

    invertidos = trabajo["_termino"].notna() & (
        trabajo["_inicio"] > trabajo["_termino"]
    )

    if invertidos.any():
        raise IntervaloInvertidoError(
            termino_col,
            tuple(trabajo.index[invertidos].tolist()),
        )

    conflictos: list[dict[str, object]] = []

    for entidad, grupo in trabajo.groupby(id_col, dropna=False):
        grupo = grupo.sort_values("_inicio").reset_index(drop=False)

        for posicion in range(len(grupo) - 1):
            actual = grupo.iloc[posicion]
            siguiente = grupo.iloc[posicion + 1]

            fin_actual = actual["_termino"]
            inicio_siguiente = siguiente["_inicio"]

            # Convención del notebook F2: ambos extremos están incluidos.
            se_solapan = (
                pd.isna(fin_actual)
                or inicio_siguiente <= fin_actual
            )

            if se_solapan:
                conflictos.append(
                    {
                        id_col: entidad,
                        "indice_a": actual["index"],
                        "indice_b": siguiente["index"],
                        "inicio_a": actual["_inicio"],
                        "termino_a": fin_actual,
                        "inicio_b": inicio_siguiente,
                        "termino_b": siguiente["_termino"],
                    }
                )

    return pd.DataFrame(conflictos)


def validar_columnas_exactas(
    df: pd.DataFrame,
    columnas_esperadas: Iterable[str],
) -> dict[str, object]:
    """Compara columnas observadas y esperadas, incluyendo el orden."""
    esperadas = list(columnas_esperadas)
    observadas = list(df.columns)

    faltantes = [columna for columna in esperadas if columna not in observadas]
    extras = [columna for columna in observadas if columna not in esperadas]

    return {
        "valido": observadas == esperadas,
        "faltantes": faltantes,
        "extras": extras,
        "orden_correcto": observadas == esperadas,
        "observadas": observadas,
        "esperadas": esperadas,
    }


def resumir_completitud(df: pd.DataFrame) -> pd.DataFrame:
    """Resume cantidad y porcentaje de nulos por columna."""
    total_filas = len(df)
    resumen = pd.DataFrame(
        {
            "columna": df.columns,
            "nulos": [int(df[columna].isna().sum()) for columna in df.columns],
        }
    )

    if total_filas == 0:
        resumen["porcentaje_nulos"] = 0.0
    else:
        resumen["porcentaje_nulos"] = resumen["nulos"] / total_filas * 100

    return resumen
