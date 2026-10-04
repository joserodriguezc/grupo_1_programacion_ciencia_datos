"""Comparación de salidas regeneradas con las versionadas."""

from pathlib import Path

import pandas as pd

# El álgebra lineal (SVD de PCA) depende de la librería BLAS/LAPACK de cada plataforma: entre
# Windows y Linux cambian los últimos dígitos (~1e-15). Diferencias mayores son un cambio real.
RTOL = 1e-9
ATOL = 1e-12


def diferencia_salida(regenerada: Path, versionada: Path) -> str | None:
    """None si ambas salidas coinciden; si no, la descripción de la diferencia.

    Exige igualdad byte a byte, salvo en CSV, donde los números de punto flotante pueden
    diferir dentro de la tolerancia. Textos, enteros, columnas y orden de filas son exactos.
    """
    if regenerada.read_bytes() == versionada.read_bytes():
        return None
    if regenerada.suffix != ".csv":
        return "los bytes difieren"
    try:
        pd.testing.assert_frame_equal(pd.read_csv(regenerada), pd.read_csv(versionada),
                                      check_exact=False, rtol=RTOL, atol=ATOL)
    except AssertionError as error:
        return str(error)
    return None
