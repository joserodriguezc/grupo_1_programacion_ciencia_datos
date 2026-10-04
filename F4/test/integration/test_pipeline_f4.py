"""Regenera la cadena de módulos de F4 (pipeline.py) en una copia temporal y la compara
con lo versionado.

Cada salida se borra de la copia antes de regenerarla: si una etapa no escribe su archivo,
el test falla en lugar de comparar el archivo versionado consigo mismo. La comparación es
byte a byte (los módulos escriben con saltos de línea LF), salvo los decimales finales que
cambian entre plataformas en el álgebra lineal (ver F4/test/utilidades.py).

Las salidas de B-Call (F4/data/results/individual/bcall) solo las produce F4_02 y se
toman de la copia como entrada; el notebook se prueba en F4/test/notebooks.
"""

import json
import shutil
from pathlib import Path

import pytest

from F4.src.analisis import conciliacion, exportacion, pipeline
from F4.src.analisis.universo import SALIDA as SALIDA_UNIVERSO
from F4.test.utilidades import diferencia_salida

RAIZ = Path(__file__).resolve().parents[3]

# Salidas regeneradas por la cadena, en orden de dependencia.
SALIDAS = {
    "auditoria": ["F4/data/reports/auditoria_entrada.json"],
    "codificacion": ["F4/data/processed/votos_codificados.csv"],
    "matrices": [
        "F4/data/processed/matriz_binaria.csv",
        "F4/data/processed/matriz_ternaria.csv",
        "F4/data/processed/matriz_nominal.csv",
        "F4/data/processed/mascara_observacion.csv",
        "F4/data/processed/afiliacion_por_votacion.csv",
    ],
    "cobertura": [
        "F4/data/reports/participacion_observada.csv",
        "F4/data/reports/celdas_sin_registro.csv",
        "F4/data/reports/conteos_partido_votacion.csv",
        "F4/data/reports/conciliacion_conteos.json",
    ],
    "cohesion": [
        "F4/data/results/partidos/cohesion_por_votacion.csv",
        "F4/data/results/partidos/resumen_cohesion.csv",
    ],
    "posicion": ["F4/data/results/partidos/posicion_partidaria.csv"],
    "afinidad": [
        "F4/data/results/pares/afinidad_diputados.csv",
        "F4/data/results/pares/matriz_hamming.csv",
        "F4/data/results/pares/hamming_partidos.csv",
    ],
    "sensibilidad": ["F4/data/reports/sensibilidad.csv"],
    "clustering": [
        "F4/data/results/grupos/clusters_diputados.csv",
        "F4/data/results/grupos/exclusiones_clustering.csv",
        "F4/data/results/grupos/matriz_hamming_clustering.csv",
        "F4/data/results/grupos/enlace_clustering.csv",
        "F4/data/results/grupos/seleccion_votaciones_clustering.csv",
        "F4/data/reports/sensibilidad_clustering.csv",
        "F4/data/reports/sensibilidad_clustering_diputados.csv",
    ],
    "pca": [
        "F4/data/results/individual/pca/coordenadas_diputados.csv",
        "F4/data/results/individual/pca/cargas_votaciones.csv",
        "F4/data/results/individual/pca/varianza_explicada.csv",
        "F4/data/results/individual/pca/exclusiones_pca.csv",
        "F4/data/results/individual/pca/seleccion_votaciones_pca.csv",
        "F4/data/results/individual/pca/medias_votaciones.csv",
    ],
    "comparacion": [
        "F4/data/results/comparacion/comparacion_clustering_bcall.csv",
        "F4/data/results/comparacion/comparacion_pca_bcall.csv",
        "F4/data/results/comparacion/correspondencia_clusters.csv",
        "F4/data/results/comparacion/resumen_comparacion.csv",
    ],
    "universo": [SALIDA_UNIVERSO],
    "conciliacion": [conciliacion.SALIDA],
}


@pytest.fixture(scope="module")
def copia_regenerada(tmp_path_factory) -> Path:
    raiz = tmp_path_factory.mktemp("repo")
    shutil.copy2(RAIZ / "pyproject.toml", raiz / "pyproject.toml")
    shutil.copytree(RAIZ / "F3/data", raiz / "F3/data")
    for carpeta in ("config", "data", "docs", "figures", "notebooks"):
        shutil.copytree(RAIZ / "F4" / carpeta, raiz / "F4" / carpeta)

    for rutas in SALIDAS.values():
        for ruta in rutas:
            (raiz / ruta).unlink()
    (raiz / exportacion.SALIDA).unlink()

    # La misma cadena que ejecuta el pipeline (D01), que además valida el contrato.
    pipeline.ejecutar(raiz)
    return raiz


@pytest.mark.slow
@pytest.mark.parametrize(
    "ruta", [ruta for rutas in SALIDAS.values() for ruta in rutas], ids=lambda r: Path(r).name
)
def test_salida_identica_a_la_versionada(copia_regenerada: Path, ruta: str) -> None:
    regenerada = copia_regenerada / ruta
    assert regenerada.is_file(), f"La cadena no escribió {ruta}"
    diferencia = diferencia_salida(regenerada, RAIZ / ruta)
    assert diferencia is None, (
        f"{ruta} difiere de la versión del repositorio: vuelva a ejecutar la cadena "
        f"o revise el cambio de método.\n{diferencia}"
    )


@pytest.mark.slow
def test_manifiesto_de_entrega_lista_todos_los_entregables(copia_regenerada: Path) -> None:
    # No se compara byte a byte: incluye hashes de CSV de PCA, que varían entre plataformas.
    manifiesto = json.loads((copia_regenerada / exportacion.SALIDA).read_text(encoding="utf-8"))
    rutas = {e["ruta"] for e in manifiesto["entregables"]}
    assert rutas == {e["ruta"] for e in exportacion.entregables(copia_regenerada)}
    assert manifiesto["protocolo"]["estado"] == "aprobado"


@pytest.mark.slow
def test_conciliacion_sin_pendientes(copia_regenerada: Path) -> None:
    conciliador, _ = conciliacion.desde_repositorio(copia_regenerada)
    reporte = conciliador.conciliar()
    assert reporte["resumen"]["pendiente"] == 0, reporte["excepciones"]
