"""Genera la big table analítica de F3 desde data/processed."""

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .integrador_dataset import construir_big_table

PROCESSED = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
)


def main() -> None:
    detalle = pd.read_csv(
        PROCESSED / "detalle_votaciones_procesado.csv",
        dtype={
            "diputado_id": "string",
            "votacion_id": "string",
        },
    )
    proyecto = pd.read_csv(
        PROCESSED / "proyecto_ley_procesado.csv",
        dtype={"Id": "string"},
    )
    diputados = pd.read_csv(
        PROCESSED / "diputados_procesados.csv",
        dtype={"diputado_id": "string"},
    )
    militancias = pd.read_csv(
        PROCESSED / "militancias_analiticas.csv",
        dtype={
            "diputado_id": "string",
            "partido_id": "string",
        },
    )

    big_table, diagnosticos = construir_big_table(
        detalle,
        proyecto,
        diputados,
        militancias,
    )

    destino = PROCESSED / "big_table_analitica.csv"
    temporal = PROCESSED / (
        f".big_table_analitica.{uuid4().hex}.tmp"
    )

    try:
        big_table.to_csv(
            temporal,
            index=False,
            encoding="utf-8",
        )
        os.replace(temporal, destino)
    finally:
        temporal.unlink(missing_ok=True)

    print(
        f"Big table: {len(big_table)} filas × "
        f"{len(big_table.columns)} columnas → {destino}"
    )
    print(
        "Diagnósticos de militancia: "
        f"{diagnosticos['estado'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()