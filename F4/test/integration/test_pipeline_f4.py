"""Regenera la cadena de módulos de F4 en una copia temporal y la compara con lo versionado.

Cada salida se borra de la copia antes de regenerarla: si una etapa no escribe su archivo,
el test falla en lugar de comparar el archivo versionado consigo mismo. La comparación es
byte a byte (los módulos escriben con saltos de línea LF).

Las salidas de B-Call (F4/data/results/individual/bcall) solo las produce F4_02 y se
toman de la copia como entrada; el notebook se prueba en F4/test/notebooks.
"""

import shutil
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from F4.src.analisis import (
    afinidad,
    auditoria,
    cobertura,
    codificacion,
    cohesion,
    conciliacion,
    matrices,
    posicion_partidos,
    sensibilidad,
)
from F4.src.analisis.agrupamiento import AgrupamientoDiputados
from F4.src.analisis.agrupamiento import guardar_resultados as guardar_agrupamiento
from F4.src.analisis.comparacion_metodos import comparar
from F4.src.analisis.pca_svd import PCASVD
from F4.src.analisis.pca_svd import guardar_resultados as guardar_pca
from F4.src.analisis.universo import SALIDA as SALIDA_UNIVERSO
from F4.src.analisis.universo import desde_repositorio as universo_desde_repositorio

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


def _ejecutar_cadena(raiz: Path) -> None:
    with (raiz / "F4/config/analisis.toml").open("rb") as archivo:
        config = tomllib.load(archivo)

    auditoria.main(raiz)
    codificacion.codificar_archivo(raiz=raiz)
    matrices.ejecutar(raiz=raiz)
    cobertura.main(raiz)
    cohesion.main(raiz)
    posicion_partidos.ejecutar(raiz=raiz)
    afinidad.ejecutar(raiz=raiz)
    sensibilidad.ejecutar(raiz=raiz)

    # Mismos parámetros que F4_04 y [sensibilidad] de analisis.toml.
    umbrales = tuple(config["sensibilidad"]["clustering_umbrales_cobertura"])
    sensibilidad.ejecutar_clustering(
        cobertura_base=0.80,
        umbrales_cobertura=umbrales,
        n_clusters=2,
        bloques=config["sensibilidad"]["clustering_bloques"],
        raiz=raiz,
    )
    nominal = afinidad.cargar_matriz_nominal(raiz / "F4/data/processed/matriz_nominal.csv")
    clusters = AgrupamientoDiputados(cobertura_minima=0.80, n_clusters=2).calcular(nominal)
    guardar_agrupamiento(clusters, raiz / "F4/data/results/grupos")

    ternaria = pd.read_csv(raiz / "F4/data/processed/matriz_ternaria.csv")
    pca = PCASVD(n_componentes=2).calcular(ternaria.set_index("diputado_id"))
    guardar_pca(pca, raiz / "F4/data/results/individual/pca")

    # Como F4_04: coordenadas PCA en memoria (releerlas del CSV redondea el último decimal).
    tablas = comparar(
        pd.read_csv(raiz / "F4/data/results/individual/bcall/bcall_diputados.csv"),
        pd.read_csv(raiz / "F4/data/results/grupos/clusters_diputados.csv"),
        pca.coordenadas,
    )
    for nombre, tabla in tablas.items():
        tabla.to_csv(raiz / "F4/data/results/comparacion" / nombre, index=False,
                     lineterminator="\n")

    universo_desde_repositorio(raiz).calcular().to_csv(
        raiz / SALIDA_UNIVERSO, index=False, lineterminator="\n")
    conciliacion.main(raiz)


@pytest.fixture(scope="module")
def copia_regenerada(tmp_path_factory) -> Path:
    raiz = tmp_path_factory.mktemp("repo")
    shutil.copy2(RAIZ / "pyproject.toml", raiz / "pyproject.toml")
    shutil.copytree(RAIZ / "F3/data", raiz / "F3/data")
    for carpeta in ("config", "data", "docs"):
        shutil.copytree(RAIZ / "F4" / carpeta, raiz / "F4" / carpeta)

    for rutas in SALIDAS.values():
        for ruta in rutas:
            (raiz / ruta).unlink()

    _ejecutar_cadena(raiz)
    return raiz


@pytest.mark.slow
@pytest.mark.parametrize(
    "ruta", [ruta for rutas in SALIDAS.values() for ruta in rutas], ids=lambda r: Path(r).name
)
def test_salida_identica_a_la_versionada(copia_regenerada: Path, ruta: str) -> None:
    regenerada = copia_regenerada / ruta
    assert regenerada.is_file(), f"La cadena no escribió {ruta}"
    assert regenerada.read_bytes() == (RAIZ / ruta).read_bytes(), (
        f"{ruta} difiere de la versión del repositorio: vuelva a ejecutar la cadena "
        "o revise el cambio de método."
    )


@pytest.mark.slow
def test_conciliacion_sin_pendientes(copia_regenerada: Path) -> None:
    conciliador, _ = conciliacion.desde_repositorio(copia_regenerada)
    reporte = conciliador.conciliar()
    assert reporte["resumen"]["pendiente"] == 0, reporte["excepciones"]
