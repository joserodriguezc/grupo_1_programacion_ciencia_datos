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
import tempfile
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from statistics import median, stdev
from time import perf_counter
from types import SimpleNamespace
from typing import Any

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


def _resumir_total_tiempo(
    etapa: str,
    tiempos: list[float],
    *,
    tipo: str,
) -> dict[str, Any]:
    """Resume un total temporal sin inventar un pico de memoria global."""

    return {
        "tipo": tipo,
        "etapa": etapa,
        "repeticiones": len(tiempos),
        "tiempo_mediana_s": median(tiempos),
        "tiempo_min_s": min(tiempos),
        "tiempo_max_s": max(tiempos),
        "tiempo_std_s": _desviacion(tiempos),
        "memoria_pico_mediana_mb": None,
        "memoria_pico_min_mb": None,
        "memoria_pico_max_mb": None,
        "memoria_pico_std_mb": None,
    }


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


def _numero_boletin_f2() -> str:
    """Obtiene el único boletín presente en el artefacto interim de F2."""

    ruta = INTERIM_DIR / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"
    boletines = (
        pd.read_csv(ruta, dtype="string")["numero_boletin"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(boletines) != 1:
        raise ValueError(
            "Se esperaba exactamente un numero_boletin en F2; "
            f"se encontraron: {boletines}"
        )

    return str(boletines[0])


class _RespuestaHTTPFixture:
    """Respuesta mínima compatible con requests.Response para F2_01."""

    def __init__(self, contenido: bytes) -> None:
        self.content = contenido

    def raise_for_status(self) -> None:
        return None


def _extraer_proyecto_f2_local(
    *,
    modulo: Any,
    numero_boletin: str,
    xml_crudo: bytes,
    xml_destino: Path,
    csv_destino: Path,
) -> pd.DataFrame:
    """Ejecuta `extraer_votaciones` de F2 sustituyendo solo la red.

    La función original conserva su lectura XML y sus escrituras, pero todas
    ellas apuntan a un sandbox temporal fuera de F2/.
    """

    xml_original = modulo.XML_FILE
    csv_original = modulo.CSV_FILE
    get_original = modulo.requests.get

    def get_local(*args: Any, **kwargs: Any) -> _RespuestaHTTPFixture:
        return _RespuestaHTTPFixture(xml_crudo)

    modulo.XML_FILE = xml_destino
    modulo.CSV_FILE = csv_destino
    modulo.requests.get = get_local

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return modulo.extraer_votaciones(numero_boletin)
    finally:
        modulo.XML_FILE = xml_original
        modulo.CSV_FILE = csv_original
        modulo.requests.get = get_original


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


def _extraer_periodos_local(
    fixture: list[dict[str, Any]],
) -> pd.DataFrame:
    _, modulo, _, _ = _modulos_extraccion()
    filas = modulo.construir_filas_periodos(fixture)

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


def _extraer_diputados_local(
    fixture: list[SimpleNamespace],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, modulo, _ = _modulos_extraccion()

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


def _extraccion_total_local(
    *,
    proyecto: Callable[[], pd.DataFrame],
    fixture_periodos: list[dict[str, Any]],
    fixture_diputados: list[SimpleNamespace],
) -> dict[str, Any]:
    diputados, militancias = _extraer_diputados_local(fixture_diputados)

    return {
        "proyecto": proyecto(),
        "periodos": _extraer_periodos_local(fixture_periodos),
        "diputados": diputados,
        "militancias": militancias,
        "detalle": _extraer_detalle_local(),
    }


def medir_extraccion_f2(
    *,
    repeticiones: int = 5,
    calentamiento: int = 1,
) -> pd.DataFrame:
    """Mide extracción F2 sin red y sin contaminar el tiempo con fixtures."""

    huella_antes = huella_f2()
    modulo_01, _, _, _ = _modulos_extraccion()

    # Preparación deliberadamente fuera de las funciones cronometradas.
    fixture_periodos = _fixture_periodos()
    fixture_diputados = _fixture_diputados()
    numero_boletin = _numero_boletin_f2()
    xml_crudo = (
        RAW_DIR / "VotacionesPorProyectoDeLey" / "proyecto_ley.xml"
    ).read_bytes()

    with tempfile.TemporaryDirectory(prefix="f3_f2_extraccion_") as temporal:
        sandbox = Path(temporal)
        xml_destino = sandbox / "raw" / "proyecto_ley.xml"
        csv_destino = sandbox / "interim" / "proyecto_ley.csv"

        def proyecto_local() -> pd.DataFrame:
            return _extraer_proyecto_f2_local(
                modulo=modulo_01,
                numero_boletin=numero_boletin,
                xml_crudo=xml_crudo,
                xml_destino=xml_destino,
                csv_destino=csv_destino,
            )

        def periodos_local() -> pd.DataFrame:
            return _extraer_periodos_local(fixture_periodos)

        def diputados_local() -> tuple[pd.DataFrame, pd.DataFrame]:
            return _extraer_diputados_local(fixture_diputados)

        def total_local() -> dict[str, Any]:
            return _extraccion_total_local(
                proyecto=proyecto_local,
                fixture_periodos=fixture_periodos,
                fixture_diputados=fixture_diputados,
            )

        etapas: list[tuple[str, Callable[[], Any]]] = [
            ("01 proyecto_ley: F2 original con respuesta local", proyecto_local),
            ("02 periodos: transformación F2", periodos_local),
            ("03 diputados/militancias: transformación F2", diputados_local),
            ("04 detalle: XML locales -> DataFrame F2", _extraer_detalle_local),
            ("TOTAL extracción reproducible sin red", total_local),
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


ESPEC_GRUPOS_PROCESAMIENTO = [
    ("configuración y carga", "## 1.", "## 4."),
    ("normalización general", "## 4.", "## 5."),
    ("procesamiento diputados", "## 5.", "## 6."),
    ("procesamiento militancias", "## 6.", "## 7."),
    ("procesamiento detalle votaciones", "## 7.", "## 8."),
    ("procesamiento proyecto ley", "## 8.", "## 9."),
    ("exportación a sandbox temporal", "## 9.", None),
]

ESPEC_GRUPOS_INTEGRACION = [
    ("configuración y carga", "## 1.", "## 3."),
    ("validaciones referenciales", "## 3.", "## 5."),
    ("integración proyecto_ley", "## 5.", "## 6."),
    ("integración diputados", "## 6.", "## 7."),
    ("integración temporal militancias", "## 7.", "## 8."),
    ("validación casos especiales", "## 8.", "## 9."),
    ("validaciones big table", "## 9.", "## 10."),
    ("diagnóstico final", "## 10.", "## 11."),
    ("exportación a sandbox temporal", "## 11.", "## 12."),
    ("criterio de término", "## 12.", None),
]


def _resolver_grupos_por_encabezados(
    ruta_notebook: Path,
    especificaciones: list[tuple[str, str, str | None]],
) -> list[tuple[str, list[int]]]:
    """Resuelve celdas de código por encabezados, sin índices rígidos.

    Si un encabezado esperado desaparece o cambia, se falla explícitamente en
    lugar de medir silenciosamente celdas equivocadas.
    """

    notebook = _leer_notebook(ruta_notebook)
    celdas = notebook["cells"]

    encabezados: list[tuple[int, str]] = []
    for indice, celda in enumerate(celdas):
        if celda.get("cell_type") != "markdown":
            continue
        texto = "".join(celda.get("source", [])).strip()
        primera = texto.splitlines()[0] if texto else ""
        if primera.startswith("## "):
            encabezados.append((indice, primera))

    def localizar(prefijo: str) -> int:
        coincidencias = [
            indice
            for indice, encabezado in encabezados
            if encabezado.startswith(prefijo)
        ]
        if len(coincidencias) != 1:
            raise RuntimeError(
                f"Encabezado {prefijo!r} en {ruta_notebook.name}: "
                f"se esperaban 1 y se encontraron {len(coincidencias)}"
            )
        return coincidencias[0]

    grupos: list[tuple[str, list[int]]] = []
    for nombre, inicio_prefijo, fin_prefijo in especificaciones:
        inicio = localizar(inicio_prefijo)
        fin = localizar(fin_prefijo) if fin_prefijo is not None else len(celdas)
        indices = [
            indice
            for indice in range(inicio + 1, fin)
            if celdas[indice].get("cell_type") == "code"
        ]
        if not indices:
            raise RuntimeError(
                f"El grupo {nombre!r} no contiene celdas de código "
                f"en {ruta_notebook.name}"
            )
        grupos.append((nombre, indices))

    return grupos


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
        _resumir_total_tiempo(
            f"TOTAL {tipo}",
            tiempos_totales,
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
        _resolver_grupos_por_encabezados(
            NOTEBOOK_PROCESAMIENTO,
            ESPEC_GRUPOS_PROCESAMIENTO,
        ),
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
        _resolver_grupos_por_encabezados(
            NOTEBOOK_INTEGRACION,
            ESPEC_GRUPOS_INTEGRACION,
        ),
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
        "etapa": "TOTAL pipeline F2 reproducible (suma descriptiva de medianas)",
        "repeticiones": int(resumen["repeticiones"].min()),
        "tiempo_mediana_s": float(resumen["tiempo_mediana_s"].sum()),
        # Extracción, procesamiento e integración se midieron en corridas
        # independientes. Sumar sus mínimos/máximos no produce extremos reales.
        "tiempo_min_s": None,
        "tiempo_max_s": None,
        "tiempo_std_s": None,
        # Los picos de memoria se reinician por etapa y no representan el pico
        # acumulado del pipeline completo.
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
