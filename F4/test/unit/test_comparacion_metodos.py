import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from F4.src.analisis.comparacion_metodos import comparar, graficar_pca  # noqa: E402


def _entradas():
    bcall = pd.DataFrame(
        {
            "diputado_id": [1, 2, 3, 4, 5, 6],
            "d1": [-0.8, -0.7, -0.6, 1.2, 1.4, np.nan],
            "grupo_bcall": ["L", "L", "L", "R", "R", "L"],
        }
    )
    # Etiquetas del clustering intercambiadas respecto de B-Call; 7 solo existe aquí.
    clusters = pd.DataFrame(
        {"diputado_id": [1, 2, 3, 4, 5, 7], "cluster": [2, 2, 2, 1, 1, 1]}
    )
    # PC1 con signo opuesto a d1; 2 no tiene coordenada.
    pca = pd.DataFrame(
        {"diputado_id": [1, 3, 4, 5, 6], "PC1": [1.5, 1.1, -2.0, -2.6, 0.3]}
    )
    return bcall, clusters, pca


def test_clusters_con_etiquetas_intercambiadas_coinciden() -> None:
    tablas = comparar(*_entradas())
    resumen = tablas["resumen_comparacion.csv"].iloc[0]
    detalle = tablas["comparacion_clustering_bcall.csv"].set_index("diputado_id")

    assert resumen["n_comunes_clustering"] == 5
    assert resumen["ari_clustering_bcall"] == pytest.approx(1)
    assert resumen["n_discrepancias_alineadas"] == 0
    assert not resumen["alineacion_clusters_ambigua"]
    assert detalle.loc[1, "cluster_alineado"] == "L"
    assert detalle.loc[4, "cluster_alineado"] == "R"


def test_detecta_discrepancias_tras_alinear() -> None:
    bcall, clusters, pca = _entradas()
    clusters.loc[clusters["diputado_id"].eq(3), "cluster"] = 1

    tablas = comparar(bcall, clusters, pca)
    resumen = tablas["resumen_comparacion.csv"].iloc[0]
    detalle = tablas["comparacion_clustering_bcall.csv"].set_index("diputado_id")

    assert resumen["n_discrepancias_alineadas"] == 1
    assert resumen["ari_clustering_bcall"] < 1
    assert not detalle.loc[3, "coincide"]


def test_alinea_el_signo_de_pc1_con_d1() -> None:
    tablas = comparar(*_entradas())
    resumen = tablas["resumen_comparacion.csv"].iloc[0]
    detalle = tablas["comparacion_pca_bcall.csv"]

    assert resumen["signo_alineacion_pc1"] == -1
    assert resumen["pearson_pc1_original_d1"] < 0
    assert resumen["pearson_pc1_alineado_d1"] == pytest.approx(
        -resumen["pearson_pc1_original_d1"]
    )
    assert resumen["spearman_pc1_alineado_d1"] == pytest.approx(1)
    np.testing.assert_allclose(detalle["PC1_alineado"], -detalle["PC1_original"])


def test_compara_pca_solo_con_d1_estimable() -> None:
    tablas = comparar(*_entradas())

    # 6 tiene PC1 pero d1 NA; 2 tiene d1 pero no PC1.
    assert set(tablas["comparacion_pca_bcall.csv"]["diputado_id"]) == {1, 3, 4, 5}
    assert tablas["resumen_comparacion.csv"].iloc[0]["n_comunes_pca"] == 4


def test_registra_universos_de_cada_metodo() -> None:
    universos = comparar(*_entradas())["universos_comparacion.csv"].set_index("diputado_id")

    assert set(universos.index) == {1, 2, 3, 4, 5, 6, 7}
    assert not universos.loc[6, "d1_disponible"]
    assert universos.loc[6, "pc1_disponible"]
    assert not universos.loc[7, "grupo_bcall_disponible"]
    assert universos.loc[7, "cluster_disponible"]
    assert not universos.loc[7, "comparado_clustering"]
    assert universos.loc[2, "comparado_clustering"]
    assert not universos.loc[2, "comparado_pca"]


def test_correspondencia_es_tabla_de_contingencia() -> None:
    correspondencia = comparar(*_entradas())["correspondencia_clusters.csv"].set_index(
        "grupo_bcall"
    )

    assert correspondencia.loc["L", 2] == 3
    assert correspondencia.loc["R", 1] == 2
    assert correspondencia.loc["L", 1] == 0


@pytest.mark.parametrize(
    "tabla, cambio, mensaje",
    [
        ("bcall", lambda t: t.drop(columns="grupo_bcall"), "faltan columnas"),
        ("clusters", lambda t: pd.concat([t, t.iloc[[0]]]), "único"),
        ("clusters", lambda t: t.assign(cluster=1), "dos grupos"),
        ("pca", lambda t: t.iloc[:2], "tres diputados comunes"),
    ],
)
def test_valida_las_entradas(tabla, cambio, mensaje) -> None:
    entradas = dict(zip(["bcall", "clusters", "pca"], _entradas()))
    entradas[tabla] = cambio(entradas[tabla])

    with pytest.raises(ValueError, match=mensaje):
        comparar(entradas["bcall"], entradas["clusters"], entradas["pca"])


def test_graficar_pca_escribe_la_figura(tmp_path) -> None:
    ruta = tmp_path / "pc1_vs_d1.png"
    graficar_pca(comparar(*_entradas())["comparacion_pca_bcall.csv"], ruta)

    assert ruta.is_file() and ruta.stat().st_size > 0
