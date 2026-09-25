"""Construcción de la línea base reproducible de la Fase 2.

Este módulo registra las entradas y salidas existentes de F2 para utilizarlas
posteriormente como referencia en las comparaciones con F3.

El baseline incluye:

- archivos raw utilizados por F2;
- tablas interim;
- productos processed;
- dimensiones de cada tabla;
- columnas y tipos de datos;
- cantidad de valores nulos;
- memoria ocupada por cada DataFrame;
- hashes SHA-256 de los archivos;
- versión de Python y pandas;
- commit Git cuando está disponible.

Este módulo no modifica los archivos de F2 ni ejecuta llamadas a servicios
externos.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

RAIZ_PROYECTO = Path(__file__).resolve().parents[3]

F2_DIR = RAIZ_PROYECTO / "F2"

RAW_DIR = F2_DIR / "data" / "raw"
INTERIM_DIR = F2_DIR / "data" / "interim"
PROCESSED_DIR = F2_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# Archivos RAW
# ---------------------------------------------------------------------------

ARCHIVOS_RAW_FIJOS = {
    "proyecto_ley_xml": (
        RAW_DIR
        / "VotacionesPorProyectoDeLey"
        / "proyecto_ley.xml"
    ),
    "diputados_periodo_10_xml": (
        RAW_DIR
        / "diputados"
        / "diputados_periodo_10.xml"
    ),
}


def obtener_xml_votaciones() -> list[Path]:
    """Obtiene los XML nominales de votaciones disponibles en F2."""

    directorio = RAW_DIR / "votaciones"

    if not directorio.exists():
        return []

    return sorted(directorio.glob("votacion_*.xml"))


# ---------------------------------------------------------------------------
# Archivos INTERIM
# ---------------------------------------------------------------------------

ARCHIVOS_INTERIM = {
    "proyecto_ley": (
        INTERIM_DIR
        / "VotacionesPorProyectoDeLey"
        / "proyecto_ley.csv"
    ),
    "diputados": INTERIM_DIR / "diputados.csv",
    "militancias": INTERIM_DIR / "militancias.csv",
    "periodos": INTERIM_DIR / "periodos.csv",
    "detalle_votaciones": (
        INTERIM_DIR
        / "votaciones"
        / "detalle_votaciones.csv"
    ),
}


# ---------------------------------------------------------------------------
# Archivos PROCESSED
# ---------------------------------------------------------------------------

ARCHIVOS_PROCESSED = {
    "diputados_procesados": (
        PROCESSED_DIR / "diputados_procesados.csv"
    ),
    "militancias_observadas": (
        PROCESSED_DIR / "militancias_observadas.csv"
    ),
    "militancias_analiticas": (
        PROCESSED_DIR / "militancias_analiticas.csv"
    ),
    "detalle_votaciones_procesado": (
        PROCESSED_DIR / "detalle_votaciones_procesado.csv"
    ),
    "proyecto_ley_procesado": (
        PROCESSED_DIR / "proyecto_ley_procesado.csv"
    ),
    "reporte_calidad": (
        PROCESSED_DIR / "reporte_calidad.csv"
    ),
    "diagnostico_integracion": (
        PROCESSED_DIR / "diagnostico_integracion.csv"
    ),
    "big_table_analitica": (
        PROCESSED_DIR / "big_table_analitica.csv"
    ),
}


# ---------------------------------------------------------------------------
# Utilidades de archivos
# ---------------------------------------------------------------------------


def verificar_archivo(ruta: Path) -> None:
    """Comprueba que un archivo requerido exista."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido para el baseline F2: {ruta}"
        )


def verificar_archivos(
    archivos: dict[str, Path],
) -> None:
    """Comprueba que todos los archivos requeridos existan."""

    faltantes = [
        ruta
        for ruta in archivos.values()
        if not ruta.exists()
    ]

    if faltantes:
        detalle = "\n".join(
            f"- {ruta}"
            for ruta in faltantes
        )

        raise FileNotFoundError(
            "Faltan archivos requeridos para construir "
            f"el baseline F2:\n{detalle}"
        )


def calcular_sha256(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    verificar_archivo(ruta)

    sha256 = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(65536), b""):
            sha256.update(bloque)

    return sha256.hexdigest()


def tamano_archivo_mb(ruta: Path) -> float:
    """Obtiene el tamaño de un archivo en MiB."""

    verificar_archivo(ruta)

    return ruta.stat().st_size / (1024**2)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def cargar_csv(
    ruta: Path,
    *,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Carga un CSV perteneciente a F2 sin modificarlo."""

    verificar_archivo(ruta)

    return pd.read_csv(
        ruta,
        dtype=dtype,
    )


def cargar_interim() -> dict[str, pd.DataFrame]:
    """Carga las tablas interim producidas por F2."""

    verificar_archivos(ARCHIVOS_INTERIM)

    return {
        nombre: cargar_csv(ruta)
        for nombre, ruta in ARCHIVOS_INTERIM.items()
    }


def cargar_processed() -> dict[str, pd.DataFrame]:
    """Carga los productos procesados utilizados como referencia F2."""

    verificar_archivos(ARCHIVOS_PROCESSED)

    return {
        nombre: cargar_csv(ruta)
        for nombre, ruta in ARCHIVOS_PROCESSED.items()
    }


# ---------------------------------------------------------------------------
# Resúmenes de tablas
# ---------------------------------------------------------------------------


def memoria_dataframe_mb(df: pd.DataFrame) -> float:
    """Calcula la memoria utilizada por un DataFrame en MiB."""

    memoria_bytes = df.memory_usage(
        index=True,
        deep=True,
    ).sum()

    return float(memoria_bytes / (1024**2))


def resumen_dataset(
    tablas: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Genera un resumen general de un conjunto de DataFrames."""

    registros: list[dict[str, Any]] = []

    for nombre, df in tablas.items():
        registros.append(
            {
                "tabla": nombre,
                "filas": len(df),
                "columnas": len(df.columns),
                "nulos": int(df.isna().sum().sum()),
                "duplicados_exactos": int(df.duplicated().sum()),
                "memoria_mb": memoria_dataframe_mb(df),
            }
        )

    return pd.DataFrame(registros)


def resumen_columnas(
    tablas: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Registra columnas, tipos, nulos y cardinalidad de cada tabla."""

    registros: list[dict[str, Any]] = []

    for nombre_tabla, df in tablas.items():
        for columna in df.columns:
            serie = df[columna]

            registros.append(
                {
                    "tabla": nombre_tabla,
                    "columna": columna,
                    "dtype": str(serie.dtype),
                    "nulos": int(serie.isna().sum()),
                    "no_nulos": int(serie.notna().sum()),
                    "valores_unicos": int(
                        serie.nunique(dropna=True)
                    ),
                }
            )

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Inventario de archivos
# ---------------------------------------------------------------------------


def inventario_archivos(
    archivos: dict[str, Path],
    etapa: str,
) -> pd.DataFrame:
    """Construye un inventario reproducible de archivos."""

    registros: list[dict[str, Any]] = []

    for nombre, ruta in archivos.items():
        verificar_archivo(ruta)

        registros.append(
            {
                "etapa": etapa,
                "nombre": nombre,
                "ruta": str(
                    ruta.relative_to(RAIZ_PROYECTO)
                ),
                "tamano_mb": tamano_archivo_mb(ruta),
                "sha256": calcular_sha256(ruta),
            }
        )

    return pd.DataFrame(registros)


def inventario_raw() -> pd.DataFrame:
    """Construye el inventario de archivos raw usados por F2."""

    archivos = dict(ARCHIVOS_RAW_FIJOS)

    for ruta in obtener_xml_votaciones():
        archivos[ruta.stem] = ruta

    return inventario_archivos(
        archivos,
        etapa="raw",
    )


def inventario_interim() -> pd.DataFrame:
    """Construye el inventario de archivos interim."""

    return inventario_archivos(
        ARCHIVOS_INTERIM,
        etapa="interim",
    )


def inventario_processed() -> pd.DataFrame:
    """Construye el inventario de productos processed."""

    return inventario_archivos(
        ARCHIVOS_PROCESSED,
        etapa="processed",
    )


# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------


def obtener_commit_git() -> str | None:
    """Obtiene el commit actual del repositorio cuando Git está disponible."""

    try:
        resultado = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=RAIZ_PROYECTO,
            capture_output=True,
            text=True,
            check=True,
        )

        return resultado.stdout.strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def obtener_entorno() -> dict[str, str | None]:
    """Registra información básica del entorno del baseline."""

    return {
        "commit_git": obtener_commit_git(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "sistema": platform.system(),
        "version_sistema": platform.release(),
        "arquitectura": platform.machine(),
    }


# ---------------------------------------------------------------------------
# Construcción del baseline
# ---------------------------------------------------------------------------


def construir_baseline_f2() -> dict[str, Any]:
    """Construye la línea base reproducible completa de F2.

    La función no modifica datos ni realiza llamadas de red.

    Returns
    -------
    dict[str, Any]
        Datos, inventarios y resúmenes necesarios para verificar
        posteriormente la equivalencia entre F2 y F3.
    """

    interim = cargar_interim()
    processed = cargar_processed()

    return {
        "entorno": obtener_entorno(),
        "interim": interim,
        "processed": processed,
        "resumen_interim": resumen_dataset(interim),
        "resumen_processed": resumen_dataset(processed),
        "columnas_interim": resumen_columnas(interim),
        "columnas_processed": resumen_columnas(processed),
        "inventario_raw": inventario_raw(),
        "inventario_interim": inventario_interim(),
        "inventario_processed": inventario_processed(),
    }