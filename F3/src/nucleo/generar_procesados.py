"""Construye las tablas procesadas de F3 desde los cinco CSV interim."""

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .generar_reportes_validacion import RUTAS
from .transformador_dataset import TransformadorDataset

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def main() -> None:
    entradas = {
        nombre: pd.read_csv(ruta, dtype="string", encoding="utf-8-sig")
        for nombre, ruta in RUTAS.items()
    }
    resultado = TransformadorDataset().transformar(entradas)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    entregables = {**resultado.tablas, "reporte_calidad": resultado.reporte_calidad}
    for nombre, tabla in entregables.items():
        destino = PROCESSED / f"{nombre}.csv"
        temporal = PROCESSED / f".{nombre}.{uuid4().hex}.tmp"
        try:
            tabla.to_csv(temporal, index=False, encoding="utf-8")
            os.replace(temporal, destino)
        finally:
            temporal.unlink(missing_ok=True)
        print(f"{nombre}: {len(tabla)} filas → {destino}")


if __name__ == "__main__":
    main()
