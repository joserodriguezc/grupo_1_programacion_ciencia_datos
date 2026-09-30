"""Congelamiento y lectura reproducible del corte analítico de F3 para F4.

Este módulo:
- lee ``F3/data/processed/big_table_analitica.csv`` sin modificarlo;
- calcula su hash SHA256;
- registra el commit Git utilizado;
- valida dimensiones y número de votaciones esperados;
- conserva el esquema observado;
- genera ``F4/data/reports/manifiesto_corte.json``.

No transforma, recodifica ni imputa los datos de F3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


RUTA_ENTRADA_POR_DEFECTO = Path("F3/data/processed/big_table_analitica.csv")
RUTA_MANIFIESTO_POR_DEFECTO = Path("F4/data/reports/manifiesto_corte.json")

FILAS_ESPERADAS = 1993
COLUMNAS_ESPERADAS = 34
VOTACIONES_ESPERADAS = 15
SEMILLA_RAIZ = 42


class ErrorCorteF4(RuntimeError):
    """Error base para fallos que impiden congelar el corte de F4."""


class ErrorValidacionCorte(ErrorCorteF4):
    """El dataset observado no satisface el contrato esperado de A01."""


@dataclass(frozen=True)
class ResumenCorte:
    """Resumen mínimo y reproducible del dataset congelado."""

    ruta: str
    sha256: str
    filas: int
    columnas: int
    votaciones: int
    votacion_id_columna: str
    esquema: dict[str, str]


def _raiz_repositorio() -> Path:
    """Obtiene la raíz del repositorio a partir de la ubicación de este módulo."""
    return Path(__file__).resolve().parents[3]


def _resolver_desde_raiz(raiz: Path, ruta: str | Path) -> Path:
    """Resuelve una ruta relativa respecto de la raíz del repositorio."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else raiz / ruta


def calcular_sha256(ruta: Path, tamano_bloque: int = 1024 * 1024) -> str:
    """Calcula SHA256 sin cargar el archivo completo en memoria."""
    digest = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(tamano_bloque), b""):
            digest.update(bloque)

    return digest.hexdigest()


def obtener_commit_git(raiz: Path) -> str:
    """Retorna el commit HEAD del repositorio."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=raiz,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ErrorCorteF4(
            "No fue posible obtener el commit Git del repositorio."
        ) from exc

    return resultado.stdout.strip()


def obtener_estado_git(raiz: Path) -> dict[str, Any]:
    """Registra rama y si existían cambios locales al crear el manifiesto."""
    try:
        rama = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=raiz,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        estado = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=raiz,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ErrorCorteF4(
            "No fue posible consultar el estado Git del repositorio."
        ) from exc

    return {
        "rama": rama or None,
        "working_tree_limpio": not bool(estado.strip()),
    }


def cargar_entrada(ruta: Path) -> pd.DataFrame:
    """Carga el CSV de F3 en modo lectura y retorna una copia en memoria."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la entrada de F3: {ruta}")

    if not ruta.is_file():
        raise ErrorCorteF4(f"La ruta de entrada no es un archivo: {ruta}")

    return pd.read_csv(ruta)


def resumir_corte(
    df: pd.DataFrame,
    ruta: Path,
    ruta_relativa: Path,
    columna_votacion: str = "votacion_id",
) -> ResumenCorte:
    """Construye el resumen verificable del corte."""
    if columna_votacion not in df.columns:
        raise ErrorValidacionCorte(
            f"Falta la columna obligatoria '{columna_votacion}'."
        )

    return ResumenCorte(
        ruta=ruta_relativa.as_posix(),
        sha256=calcular_sha256(ruta),
        filas=int(df.shape[0]),
        columnas=int(df.shape[1]),
        votaciones=int(df[columna_votacion].nunique(dropna=True)),
        votacion_id_columna=columna_votacion,
        esquema={columna: str(tipo) for columna, tipo in df.dtypes.items()},
    )


def validar_corte(
    resumen: ResumenCorte,
    *,
    filas_esperadas: int = FILAS_ESPERADAS,
    columnas_esperadas: int = COLUMNAS_ESPERADAS,
    votaciones_esperadas: int = VOTACIONES_ESPERADAS,
) -> list[str]:
    """Valida A01 y retorna una lista de discrepancias.

    No modifica el dataset. La decisión de aceptar una variación debe quedar
    documentada fuera de esta función.
    """
    discrepancias: list[str] = []

    if resumen.filas != filas_esperadas:
        discrepancias.append(
            f"filas: observadas={resumen.filas}, esperadas={filas_esperadas}"
        )

    if resumen.columnas != columnas_esperadas:
        discrepancias.append(
            "columnas: "
            f"observadas={resumen.columnas}, esperadas={columnas_esperadas}"
        )

    if resumen.votaciones != votaciones_esperadas:
        discrepancias.append(
            "votaciones: "
            f"observadas={resumen.votaciones}, esperadas={votaciones_esperadas}"
        )

    return discrepancias


def construir_manifiesto(
    *,
    raiz: Path,
    resumen: ResumenCorte,
    discrepancias: list[str],
    semilla: int = SEMILLA_RAIZ,
    justificacion_variacion: str | None = None,
) -> dict[str, Any]:
    """Construye el contenido serializable del manifiesto A01."""
    if discrepancias and not justificacion_variacion:
        raise ErrorValidacionCorte(
            "El corte difiere de lo esperado y no se proporcionó una "
            "justificación. Discrepancias: "
            + "; ".join(discrepancias)
        )

    estado_git = obtener_estado_git(raiz)

    return {
        "fase": "F4",
        "tarea": "A01",
        "nombre": "Congelar la entrada analítica",
        "generado_en_utc": datetime.now(UTC).isoformat(),
        "entrada": asdict(resumen),
        "esperado": {
            "filas": FILAS_ESPERADAS,
            "columnas": COLUMNAS_ESPERADAS,
            "votaciones": VOTACIONES_ESPERADAS,
        },
        "validacion": {
            "estado": "ok" if not discrepancias else "variacion_justificada",
            "discrepancias": discrepancias,
            "justificacion_variacion": justificacion_variacion,
        },
        "reproducibilidad": {
            "commit_git": obtener_commit_git(raiz),
            "rama_git": estado_git["rama"],
            "working_tree_limpio": estado_git["working_tree_limpio"],
            "semilla_raiz": semilla,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "plataforma": platform.platform(),
        },
        "reglas": {
            "entrada_solo_lectura": True,
            "copia_dato_crudo_en_f4": False,
            "transformaciones_aplicadas": False,
            "imputaciones_aplicadas": False,
        },
    }


def guardar_manifiesto(manifiesto: dict[str, Any], ruta_salida: Path) -> None:
    """Escribe el manifiesto JSON de forma determinista y legible."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with ruta_salida.open("w", encoding="utf-8") as archivo:
        json.dump(
            manifiesto,
            archivo,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        archivo.write("\n")


def congelar_corte(
    *,
    ruta_entrada: str | Path = RUTA_ENTRADA_POR_DEFECTO,
    ruta_manifiesto: str | Path = RUTA_MANIFIESTO_POR_DEFECTO,
    semilla: int = SEMILLA_RAIZ,
    justificacion_variacion: str | None = None,
    raiz: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ejecuta A01: carga, identifica, valida y documenta el corte de F3."""
    raiz = (raiz or _raiz_repositorio()).resolve()
    entrada = _resolver_desde_raiz(raiz, ruta_entrada)
    salida = _resolver_desde_raiz(raiz, ruta_manifiesto)

    try:
        entrada_relativa = entrada.relative_to(raiz)
    except ValueError:
        entrada_relativa = entrada

    df = cargar_entrada(entrada)
    resumen = resumir_corte(df, entrada, entrada_relativa)
    discrepancias = validar_corte(resumen)

    manifiesto = construir_manifiesto(
        raiz=raiz,
        resumen=resumen,
        discrepancias=discrepancias,
        semilla=semilla,
        justificacion_variacion=justificacion_variacion,
    )

    guardar_manifiesto(manifiesto, salida)

    return df, manifiesto


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Congela y documenta la entrada analítica de F3 para F4 (A01)."
    )
    parser.add_argument(
        "--entrada",
        default=str(RUTA_ENTRADA_POR_DEFECTO),
        help="Ruta del CSV de F3 relativa a la raíz del repositorio.",
    )
    parser.add_argument(
        "--salida",
        default=str(RUTA_MANIFIESTO_POR_DEFECTO),
        help="Ruta del manifiesto JSON relativa a la raíz del repositorio.",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=SEMILLA_RAIZ,
        help="Semilla raíz registrada para reproducibilidad.",
    )
    parser.add_argument(
        "--justificacion-variacion",
        default=None,
        help="Justificación obligatoria si cambian filas, columnas o votaciones.",
    )
    return parser


def main() -> int:
    """Punto de entrada para ejecución desde consola."""
    args = _crear_parser().parse_args()

    try:
        _, manifiesto = congelar_corte(
            ruta_entrada=args.entrada,
            ruta_manifiesto=args.salida,
            semilla=args.semilla,
            justificacion_variacion=args.justificacion_variacion,
        )
    except (ErrorCorteF4, FileNotFoundError) as exc:
        print(f"ERROR A01: {exc}", file=sys.stderr)
        return 1

    entrada = manifiesto["entrada"]
    print(
        "A01 completado: "
        f"{entrada['filas']} filas, "
        f"{entrada['columnas']} columnas, "
        f"{entrada['votaciones']} votaciones."
    )
    print(f"SHA256: {entrada['sha256']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
