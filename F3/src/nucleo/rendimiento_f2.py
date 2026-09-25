"""Benchmark reproducible y seguro de la línea base F2.

Este módulo mide extracción, procesamiento e integración de F2 manteniendo
`F2/` estrictamente como fuente de solo lectura.

Principios de diseño
--------------------
1. Nunca escribe dentro de `F2/`.
2. Extracción usa artefactos locales existentes y funciones de transformación
   originales cuando es posible; no llama APIs ni SOAP.
3. Procesamiento e integración se ejecutan sobre una copia temporal de los
   datos de F2. Si una celda original hace `to_csv`, escribe únicamente dentro
   del sandbox temporal, que se elimina al terminar la repetición.
4. Antes y después de cada benchmark se calcula una huella SHA-256 del árbol
   `F2/`. Si cambia un archivo, se lanza un error.
5. Tiempo y memoria se reportan por etapa y total del bloque.

La medición de memoria usa `tracemalloc`, por lo que puede no capturar toda la
memoria nativa empleada por NumPy/pandas.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import tracemalloc
from pathlib import Path
from statistics import median, stdev
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

from F3.src.nucleo.metricas import benchmark


RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
F2_DIR = RAIZ_PROYECTO / "F2"
RAW_DIR = F2_DIR / "data" / "raw"
INTERIM_DIR = F2_DIR / "data" / "interim"
PROCESSED_DIR = F2_DIR / "data" / "processed"

NOTEBOOK_PROCESAMIENTO = F2_DIR / "notebooks" / "F2_03_procesamiento_validacion.ipynb"
NOTEBOOK_INTEGRACION = F2_DIR / "notebooks" / "F2_04_Integración.ipynb"


# ---------------------------------------------------------------------------
# Protección de la línea base F2
# ---------------------------------------------------------------------------


def _sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)

    return digest.hexdigest()


def huella_f2() -> dict[str, str]:
    """Devuelve una huella de todos los archivos versionables de F2.

    Se ignoran artefactos de ejecución de Python (`__pycache__`, `.pyc`) porque
    pueden aparecer al importar módulos sin representar un cambio de la línea
    base de datos/código fuente.
    """

    huella: dict[str, str] = {}

    for ruta in sorted(F2_DIR.rglob("*")):
        if not ruta.is_file():
            continue

        if "__pycache__" in ruta.parts or ruta.suffix == ".pyc":
            continue

        relativa = str(ruta.relative_to(F2_DIR))
        huella[relativa] = _sha256_archivo(ruta)

    return huella


def _verificar_huella_f2(antes: dict[str, str], contexto: str) -> None:
    despues = huella_f2()

    if antes == despues:
        return

    claves = sorted(set(antes) | set(despues))
    cambios = [
        clave
        for clave in claves
        if antes.get(clave) != despues.get(clave)
    ]

    detalle = "\n".join(f"- {ruta}" for ruta in cambios[:20])
    raise RuntimeError(
        "La línea base F2 cambió durante el benchmark "
        f"({contexto}). Archivos distintos:\n{detalle}"
    )


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un módulo Python desde una ruta cuyo nombre puede iniciar en número."""

    spec = importlib.util.spec_from_file_location(nombre, ruta)

    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo: {ruta}")

    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _desviacion(valores: list[float]) -> float:
    return stdev(valores) if len(valores) > 1 else 0.0


def _resumir_repeticiones(
    etapa: str,
    tiempos: list[float],
    memorias: list[float],
    *,
    tipo: str,
    filas_salida: int | None = None,
) -> dict[str, Any]:
    fila: dict[str, Any] = {
        "tipo": tipo,
        "etapa": etapa,
        "repeticiones": len(tiempos),
        "tiempo_mediana_s": median(tiempos),
        "tiempo_min_s": min(tiempos),
        "tiempo_max_s": max(tiempos),
        "tiempo_std_s": _desviacion(tiempos),
        "memoria_pico_mediana_mb": median(memorias),
        "memoria_pico_min_mb": min(memorias),
        "memoria_pico_max_mb": max(memorias),
        "memoria_pico_std_mb": _desviacion(memorias),
    }

    if filas_salida is not None:
        fila["filas_salida"] = filas_salida

    return fila


def _filas_resultado(resultado: Any) -> int | None:
    if isinstance(resultado, pd.DataFrame):
        return len(resultado)

    if isinstance(resultado, tuple):
        conteos = [
            len(elemento)
            for elemento in resultado
            if isinstance(elemento, pd.DataFrame)
        ]
        if conteos:
            return sum(conteos)

    if isinstance(resultado, dict):
        conteos = [
            len(elemento)
            for elemento in resultado.values()
            if isinstance(elemento, pd.DataFrame)
        ]
        if conteos:
            return sum(conteos)

    return None


def _leer_notebook(ruta: Path) -> dict[str, Any]:
    with ruta.open(encoding="utf-8") as archivo:
        return json.load(archivo)


# ---------------------------------------------------------------------------
# Sandbox temporal: toda escritura queda fuera de F2/
# ---------------------------------------------------------------------------


def _copiar_directorio(origen: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(origen, destino, dirs_exist_ok=True)


def _crear_sandbox_procesamiento(base_temporal: Path) -> Path:
    """Crea F2/data/interim temporal para ejecutar F2_03 de forma aislada."""

    sandbox_f2 = base_temporal / "F2"
    _copiar_directorio(INTERIM_DIR, sandbox_f2 / "data" / "interim")
    (sandbox_f2 / "data" / "processed").mkdir(parents=True, exist_ok=True)
    return sandbox_f2


def _crear_sandbox_integracion(base_temporal: Path) -> Path:
    """Crea F2/data/processed temporal para ejecutar F2_04 de forma aislada."""

    sandbox_f2 = base_temporal / "F2"
    _copiar_directorio(PROCESSED_DIR, sandbox_f2 / "data" / "processed")
    return sandbox_f2


# ---------------------------------------------------------------------------
# 4.1 Extracción F2: transformación local sin red
# ---------------------------------------------------------------------------


_MOD_01 = None
_MOD_02 = None
_MOD_03 = None
_MOD_04 = None


def _modulos_extraccion():
    global _MOD_01, _MOD_02, _MOD_03, _MOD_04

    if _MOD_01 is None:
        _MOD_01 = _cargar_modulo(
            "f2_01_extraer_votaciones",
            F2_DIR / "src" / "01_extraer_votaciones_proyecto.py",
        )
        _MOD_02 = _cargar_modulo(
            "f2_02_periodos",
            F2_DIR / "src" / "02_periodos_legislativos.py",
        )
        _MOD_03 = _cargar_modulo(
            "f2_03_diputados",
            F2_DIR / "src" / "03_diputados.py",
        )
        _MOD_04 = _cargar_modulo(
            "f2_04_detalle",
            F2_DIR / "src" / "04_extraer_detalles_votaciones.py",
        )

    return _MOD_01, _MOD_02, _MOD_03, _MOD_04


def _extraer_proyecto_local() -> pd.DataFrame:
    ruta_xml = RAW_DIR / "VotacionesPorProyectoDeLey" / "proyecto_ley.xml"

    df = pd.read_xml(
        ruta_xml,
        xpath=".//*[local-name()='VotacionProyectoLey']",
    )

    if not df.empty:
        df.insert(0, "numero_boletin", "11092-07")

    return df


def _fixture_periodos() -> list[dict[str, Any]]:
    df = pd.read_csv(INTERIM_DIR / "periodos.csv")

    return [
        {
            "Id": fila.periodo_id,
            "Nombre": fila.nombre,
            "FechaInicio": fila.fecha_inicio,
            "FechaTermino": fila.fecha_termino,
        }
        for fila in df.itertuples(index=False)
    ]


def _extraer_periodos_local() -> pd.DataFrame:
    _, modulo, _, _ = _modulos_extraccion()
    filas = modulo.construir_filas_periodos(_fixture_periodos())

    return (
        pd.DataFrame(filas)
        .sort_values("fecha_inicio")
        .reset_index(drop=True)
    )


def _valor_o_none(valor: Any) -> Any:
    return None if pd.isna(valor) else valor


def _fixture_diputados() -> list[SimpleNamespace]:
    diputados = pd.read_csv(INTERIM_DIR / "diputados.csv", dtype=str)
    militancias = pd.read_csv(INTERIM_DIR / "militancias.csv", dtype=str)

    militancias_por_diputado = {
        str(diputado_id): grupo
        for diputado_id, grupo in militancias.groupby("diputado_id", dropna=False)
    }

    salida: list[SimpleNamespace] = []

    for fila in diputados.itertuples(index=False):
        diputado_id = str(fila.diputado_id)
        filas_militancia = militancias_por_diputado.get(diputado_id)
        historial = []

        if filas_militancia is not None:
            for militancia in filas_militancia.itertuples(index=False):
                historial.append(
                    SimpleNamespace(
                        Partido=SimpleNamespace(
                            Id=_valor_o_none(militancia.partido_id),
                            Nombre=_valor_o_none(militancia.partido_nombre),
                            Alias=_valor_o_none(militancia.partido_alias),
                        ),
                        FechaInicio=_valor_o_none(militancia.fecha_inicio),
                        FechaTermino=_valor_o_none(militancia.fecha_termino),
                    )
                )

        salida.append(
            SimpleNamespace(
                Diputado=SimpleNamespace(
                    Id=_valor_o_none(fila.diputado_id),
                    Nombre=_valor_o_none(fila.nombre),
                    Nombre2=_valor_o_none(fila.nombre2),
                    ApellidoPaterno=_valor_o_none(fila.apellido_paterno),
                    ApellidoMaterno=_valor_o_none(fila.apellido_materno),
                    FechaNacimiento=_valor_o_none(fila.fecha_nacimiento),
                    RUT=_valor_o_none(fila.rut),
                    RUTDV=_valor_o_none(fila.rut_dv),
                    Sexo={
                        "Valor": _valor_o_none(fila.sexo_valor),
                        "_value_1": _valor_o_none(fila.sexo_desc),
                    },
                    Militancias=SimpleNamespace(Militancia=historial),
                ),
                FechaInicio=_valor_o_none(fila.fecha_inicio_periodo),
                FechaTermino=_valor_o_none(fila.fecha_termino_periodo),
            )
        )

    return salida


def _extraer_diputados_local() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, modulo, _ = _modulos_extraccion()
    fixture = _fixture_diputados()

    diputados = pd.DataFrame(
        modulo.construir_filas_diputados(fixture, "10")
    )
    militancias = pd.DataFrame(
        modulo.construir_filas_militancias(fixture)
    )

    return diputados, militancias


def _extraer_detalle_local() -> pd.DataFrame:
    _, _, _, modulo = _modulos_extraccion()
    archivos = sorted((RAW_DIR / "votaciones").glob("votacion_*.xml"))

    with contextlib.redirect_stdout(io.StringIO()):
        return modulo.consolidar_xml_votaciones(archivos)


def _extraccion_total_local() -> dict[str, Any]:
    proyecto = _extraer_proyecto_local()
    periodos = _extraer_periodos_local()
    diputados, militancias = _extraer_diputados_local()
    detalle = _extraer_detalle_local()

    return {
        "proyecto": proyecto,
        "periodos": periodos,
        "diputados": diputados,
        "militancias": militancias,
        "detalle": detalle,
    }


def medir_extraccion_f2(
    *,
    repeticiones: int = 5,
    calentamiento: int = 1,
) -> pd.DataFrame:
    """Mide transformación local de extracción sin llamadas ni escrituras."""

    huella_antes = huella_f2()

    etapas: list[tuple[str, Callable[[], Any]]] = [
        ("01 proyecto_ley XML -> DataFrame", _extraer_proyecto_local),
        ("02 periodos: transformación local", _extraer_periodos_local),
        ("03 diputados/militancias: transformación local", _extraer_diputados_local),
        ("04 detalle: XML locales -> DataFrame", _extraer_detalle_local),
        ("TOTAL extracción reproducible sin red", _extraccion_total_local),
    ]

    filas = []

    for nombre, funcion in etapas:
        resultado, metricas = benchmark(
            funcion,
            repeticiones=repeticiones,
            calentamiento=calentamiento,
        )

        filas.append(
            {
                "tipo": "extraccion",
                "etapa": nombre,
                **metricas,
                "filas_salida": _filas_resultado(resultado),
            }
        )

    _verificar_huella_f2(huella_antes, "extracción")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Ejecución original de notebooks dentro del sandbox
# ---------------------------------------------------------------------------


GRUPOS_PROCESAMIENTO = [
    ("configuración y carga", [2, 3, 5, 7]),
    ("normalización general", [9, 11, 13, 15, 17, 18, 20]),
    ("procesamiento diputados", [23, 25, 27, 29]),
    ("procesamiento militancias", [31, 33, 34, 36, 37, 38, 39, 40, 42]),
    ("procesamiento detalle votaciones", [44, 46, 48]),
    ("procesamiento proyecto ley", [50, 52, 54, 56, 57]),
    ("exportación a sandbox temporal", [59, 60, 62]),
]

GRUPOS_INTEGRACION = [
    ("configuración y carga", [2, 3, 5]),
    ("validaciones referenciales", [7]),
    ("integración proyecto_ley", [10]),
    ("integración diputados", [12]),
    ("integración temporal militancias", [14, 15]),
    ("validación casos especiales", [17]),
    ("validaciones big table", [19]),
    ("diagnóstico final", [21]),
    ("exportación a sandbox temporal", [23]),
    ("criterio de término", [25]),
]


def _ejecutar_grupos_una_vez(
    ruta_notebook: Path,
    grupos: list[tuple[str, list[int]]],
    sandbox_root: Path,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    notebook = _leer_notebook(ruta_notebook)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    resultados: list[dict[str, float]] = []

    cwd_anterior = Path.cwd()

    try:
        os.chdir(sandbox_root)

        for nombre, indices in grupos:
            tracemalloc.start()
            inicio = perf_counter()

            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    for indice in indices:
                        celda = notebook["cells"][indice]

                        if celda["cell_type"] != "code":
                            continue

                        codigo = "".join(celda.get("source", []))

                        exec(
                            compile(
                                codigo,
                                f"{ruta_notebook.name}:cell_{indice}",
                                "exec",
                            ),
                            namespace,
                            namespace,
                        )

                tiempo = perf_counter() - inicio
                _, memoria_pico = tracemalloc.get_traced_memory()

            finally:
                tracemalloc.stop()

            resultados.append(
                {
                    "etapa": nombre,
                    "tiempo_s": tiempo,
                    "memoria_pico_mb": memoria_pico / (1024**2),
                }
            )

    finally:
        os.chdir(cwd_anterior)

    return resultados, namespace


def _benchmark_notebook_sandbox(
    ruta_notebook: Path,
    grupos: list[tuple[str, list[int]]],
    *,
    tipo: str,
    preparar_sandbox: Callable[[Path], Path],
    repeticiones: int,
    calentamiento: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser mayor o igual a 1")

    if calentamiento < 0:
        raise ValueError("calentamiento no puede ser negativo")

    # Cada calentamiento obtiene un sandbox nuevo.
    for _ in range(calentamiento):
        with tempfile.TemporaryDirectory(prefix="f3_f2_benchmark_") as temporal:
            raiz_temporal = Path(temporal)
            preparar_sandbox(raiz_temporal)
            _ejecutar_grupos_una_vez(
                ruta_notebook,
                grupos,
                raiz_temporal,
            )

    tiempos_por_etapa = {nombre: [] for nombre, _ in grupos}
    memorias_por_etapa = {nombre: [] for nombre, _ in grupos}
    tiempos_totales: list[float] = []
    memorias_totales: list[float] = []
    ultimo_namespace: dict[str, Any] = {}

    for _ in range(repeticiones):
        with tempfile.TemporaryDirectory(prefix="f3_f2_benchmark_") as temporal:
            raiz_temporal = Path(temporal)
            preparar_sandbox(raiz_temporal)

            resultados, namespace = _ejecutar_grupos_una_vez(
                ruta_notebook,
                grupos,
                raiz_temporal,
            )

            tiempos_totales.append(
                sum(fila["tiempo_s"] for fila in resultados)
            )
            memorias_totales.append(
                max(fila["memoria_pico_mb"] for fila in resultados)
            )
            ultimo_namespace = namespace

            for fila in resultados:
                nombre = fila["etapa"]
                tiempos_por_etapa[nombre].append(fila["tiempo_s"])
                memorias_por_etapa[nombre].append(fila["memoria_pico_mb"])

    filas = [
        _resumir_repeticiones(
            nombre,
            tiempos_por_etapa[nombre],
            memorias_por_etapa[nombre],
            tipo=tipo,
        )
        for nombre, _ in grupos
    ]

    filas.append(
        _resumir_repeticiones(
            f"TOTAL {tipo}",
            tiempos_totales,
            memorias_totales,
            tipo=tipo,
        )
    )

    return pd.DataFrame(filas), ultimo_namespace


# ---------------------------------------------------------------------------
# 4.2 Procesamiento
# ---------------------------------------------------------------------------


def medir_procesamiento_f2(
    *,
    repeticiones: int = 5,
    calentamiento: int = 1,
) -> pd.DataFrame:
    """Mide F2_03 dentro de un sandbox; F2 original permanece intacta."""

    huella_antes = huella_f2()

    resultados, namespace = _benchmark_notebook_sandbox(
        NOTEBOOK_PROCESAMIENTO,
        GRUPOS_PROCESAMIENTO,
        tipo="procesamiento",
        preparar_sandbox=_crear_sandbox_procesamiento,
        repeticiones=repeticiones,
        calentamiento=calentamiento,
    )

    salidas = {
        "procesamiento diputados": namespace.get("diputados_procesados"),
        "procesamiento militancias": namespace.get("militancias_analiticas"),
        "procesamiento detalle votaciones": namespace.get("detalle_votaciones_procesado"),
        "procesamiento proyecto ley": namespace.get("proyecto_ley_procesado"),
        "exportación a sandbox temporal": namespace.get("reporte_calidad_df"),
    }

    for etapa, df in salidas.items():
        if isinstance(df, pd.DataFrame):
            resultados.loc[
                resultados["etapa"] == etapa,
                "filas_salida",
            ] = len(df)

    _verificar_huella_f2(huella_antes, "procesamiento")
    return resultados


# ---------------------------------------------------------------------------
# 4.3 Integración
# ---------------------------------------------------------------------------


def medir_integracion_f2(
    *,
    repeticiones: int = 5,
    calentamiento: int = 1,
) -> pd.DataFrame:
    """Mide F2_04 dentro de un sandbox; F2 original permanece intacta."""

    huella_antes = huella_f2()

    resultados, namespace = _benchmark_notebook_sandbox(
        NOTEBOOK_INTEGRACION,
        GRUPOS_INTEGRACION,
        tipo="integracion",
        preparar_sandbox=_crear_sandbox_integracion,
        repeticiones=repeticiones,
        calentamiento=calentamiento,
    )

    big_table = namespace.get("big_table")
    big_table_exportar = namespace.get("big_table_exportar")

    if isinstance(big_table, pd.DataFrame):
        resultados.loc[
            resultados["etapa"] == "integración temporal militancias",
            "filas_salida",
        ] = len(big_table)

    if isinstance(big_table_exportar, pd.DataFrame):
        resultados.loc[
            resultados["etapa"] == "exportación a sandbox temporal",
            "filas_salida",
        ] = len(big_table_exportar)

    _verificar_huella_f2(huella_antes, "integración")
    return resultados


# ---------------------------------------------------------------------------
# 4.4 Resumen del pipeline
# ---------------------------------------------------------------------------


def resumir_pipeline_f2(
    resultados_extraccion: pd.DataFrame,
    resultados_procesamiento: pd.DataFrame,
    resultados_integracion: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida los resultados ya medidos sin volver a ejecutar F2."""

    total_extraccion = resultados_extraccion[
        resultados_extraccion["etapa"] == "TOTAL extracción reproducible sin red"
    ].copy()

    total_procesamiento = resultados_procesamiento[
        resultados_procesamiento["etapa"] == "TOTAL procesamiento"
    ].copy()

    total_integracion = resultados_integracion[
        resultados_integracion["etapa"] == "TOTAL integracion"
    ].copy()

    resumen = pd.concat(
        [
            total_extraccion,
            total_procesamiento,
            total_integracion,
        ],
        ignore_index=True,
    )

    total_pipeline = {
        "tipo": "pipeline",
        "etapa": "TOTAL pipeline F2 reproducible",
        "repeticiones": int(resumen["repeticiones"].min()),
        "tiempo_mediana_s": float(resumen["tiempo_mediana_s"].sum()),
        "tiempo_min_s": float(resumen["tiempo_min_s"].sum()),
        "tiempo_max_s": float(resumen["tiempo_max_s"].sum()),
        "tiempo_std_s": None,
        "memoria_pico_mediana_mb": None,
        "memoria_pico_min_mb": None,
        "memoria_pico_max_mb": None,
        "memoria_pico_std_mb": None,
        "filas_salida": None,
    }

    return pd.concat(
        [resumen, pd.DataFrame([total_pipeline])],
        ignore_index=True,
    )
