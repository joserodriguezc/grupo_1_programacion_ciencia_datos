"""Conserva los reportes de calidad de F3 como evidencia JSON."""

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .registro_validacion import ReporteValidacion

DIRECTORIO_REPORTES = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "reports"
    / "validaciones"
)


def _huella_sha256(ruta: Path) -> str:
    digest = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b"",
        ):
            digest.update(bloque)

    return digest.hexdigest()


def guardar_reporte_json(
    reporte: ReporteValidacion,
    *,
    directorio: Path | str = DIRECTORIO_REPORTES,
    archivo_entrada: Path | str | None = None,
    archivos_entrada: Mapping[str, Path | str] | None = None,
) -> Path:
    """Guarda un reporte por ejecución y devuelve su ruta."""
    origen = (
        Path(archivo_entrada)
        if archivo_entrada is not None
        else None
    )
    huella = (
        _huella_sha256(origen)
        if origen is not None
        else None
    )
    fecha = datetime.now(UTC)
    if archivo_entrada is not None and archivos_entrada is not None:
        raise ValueError(
            "Indica archivo_entrada o archivos_entrada, no ambos"
        )

    entradas = None
    if archivos_entrada is not None:
        if not archivos_entrada:
            raise ValueError("archivos_entrada no puede estar vacío")

        entradas = {
            nombre: {
                "ruta": str(ruta),
                "sha256": _huella_sha256(Path(ruta)),
            }
            for nombre, ruta in archivos_entrada.items()
        }    
        
    contenido = {
        "generado_utc": fecha.isoformat(),
        "tabla": reporte.tabla,
        "aprobado": reporte.aprobado,
        "archivo_entrada": (
            str(origen) if origen is not None else None
        ),
        "sha256_entrada": huella,
        "registros": [
            asdict(registro)
            for registro in reporte.registros
        ],
    }
    
    if entradas is not None:
        contenido["archivos_entrada"] = entradas
    carpeta = Path(directorio)
    carpeta.mkdir(parents=True, exist_ok=True)

    nombre_tabla = (
        re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            reporte.tabla,
        ).strip("_")
        or "tabla"
    )
    
    nombre = f"{nombre_tabla}.json"
    destino = carpeta / nombre
    temporal = carpeta / f".{nombre}.{uuid4().hex}.tmp"

    try:
        with temporal.open("x", encoding="utf-8") as archivo:
            json.dump(
                contenido,
                archivo,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            archivo.write("\n")

        os.replace(temporal, destino)
    finally:
        temporal.unlink(missing_ok=True)

    return destino