import sys
from pathlib import Path

import pandas as pd

SRC_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

from validaciones import detectar_solapamientos_vigencia  # noqa: E402


def test_no_existen_solapamientos(
    militancias_analiticas,
):
    conflictos = detectar_solapamientos_vigencia(
        militancias_analiticas,
    )

    assert conflictos.empty, (
        "Se detectaron solapamientos de militancia:\n"
        + conflictos.head(10).to_string(index=False)
    )


def test_intervalos_validos(
    militancias_analiticas,
):
    df = militancias_analiticas.copy()

    inicio = pd.to_datetime(
        df["fecha_inicio"],
        errors="coerce",
    )

    termino = pd.to_datetime(
        df["fecha_termino"],
        errors="coerce",
    )

    invalidos = df[
        termino.notna()
        & (inicio > termino)
    ]

    assert invalidos.empty, (
        "Existen militancias con fecha_inicio "
        "posterior a fecha_termino."
    )


def test_partido_alias_completo(
    militancias_analiticas,
):
    alias = (
        militancias_analiticas[
            "partido_alias"
        ]
        .astype("string")
        .str.strip()
    )

    invalidos = alias.isna() | alias.eq("")

    assert not invalidos.any(), (
        "partido_alias contiene valores nulos o vacíos."
    )