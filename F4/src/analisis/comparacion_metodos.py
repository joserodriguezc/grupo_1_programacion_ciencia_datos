"""Comparación descriptiva de clustering y PCA con B-Call."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .sensibilidad import _comparar_clusters


def _indexar(tabla, columnas, nombre):
    requeridas = {"diputado_id", *columnas}
    faltantes = requeridas - set(tabla.columns)

    if faltantes:
        raise ValueError(
            f"{nombre}: faltan columnas {sorted(faltantes)}."
        )

    if (
        tabla["diputado_id"].isna().any()
        or tabla["diputado_id"].duplicated().any()
    ):
        raise ValueError(
            f"{nombre}: diputado_id debe ser único y no nulo."
        )

    return tabla.set_index("diputado_id").copy()


def comparar(bcall, clusters, pca):
    """Compara salidas existentes sin volver a estimar B-Call."""
    b = _indexar(bcall, ["d1", "grupo_bcall"], "B-Call")
    c = _indexar(clusters, ["cluster"], "Clustering")
    p = _indexar(pca, ["PC1"], "PCA")

    b["d1"] = pd.to_numeric(b["d1"], errors="raise")
    p["PC1"] = pd.to_numeric(p["PC1"], errors="raise")

    b_grupos = b.loc[
        b["grupo_bcall"].notna(), "grupo_bcall"
    ]
    c_grupos = c.loc[c["cluster"].notna(), "cluster"]

    if b_grupos.nunique() != 2 or c_grupos.nunique() != 2:
        raise ValueError(
            "Esta comparación requiere dos grupos en cada método."
        )

    comunes_c = b_grupos.index.intersection(
        c_grupos.index, sort=False
    )

    if len(comunes_c) < 2:
        raise ValueError(
            "Clustering y B-Call tienen menos de dos diputados comunes."
        )

    ari, mapa, ambiguo = _comparar_clusters(
        b_grupos.loc[comunes_c],
        c_grupos.loc[comunes_c],
    )

    detalle_c = pd.DataFrame(
        {
            "grupo_bcall": b_grupos.loc[comunes_c],
            "cluster": c_grupos.loc[comunes_c],
        }
    )
    detalle_c["cluster_alineado"] = detalle_c["cluster"].map(mapa)
    detalle_c["coincide"] = detalle_c["grupo_bcall"].eq(
        detalle_c["cluster_alineado"]
    )
    detalle_c["alineacion_ambigua"] = ambiguo

    correspondencia = pd.crosstab(
        detalle_c["grupo_bcall"],
        detalle_c["cluster"],
    )

    # Comparar únicamente coordenadas y puntajes estimables.
    b_validos = b.index[np.isfinite(b["d1"])]
    p_validos = p.index[np.isfinite(p["PC1"])]

    comunes_p = b_validos.intersection(
        p_validos, sort=False
    )

    if len(comunes_p) < 3:
        raise ValueError(
            "PCA y B-Call requieren al menos tres diputados comunes."
        )

    detalle_p = pd.DataFrame(
        {
            "PC1_original": p.loc[comunes_p, "PC1"],
            "d1": b.loc[comunes_p, "d1"],
            "grupo_bcall": b.loc[comunes_p, "grupo_bcall"],
        }
    )

    if (
        detalle_p["PC1_original"].nunique() < 2
        or detalle_p["d1"].nunique() < 2
    ):
        raise ValueError(
            "La correlación requiere variación en PC1 y d1."
        )

    r_original = detalle_p["PC1_original"].corr(
        detalle_p["d1"]
    )
    signo = -1 if r_original < 0 else 1

    detalle_p["signo_alineacion"] = signo
    detalle_p["PC1_alineado"] = (
        signo * detalle_p["PC1_original"]
    )

    pearson = detalle_p["PC1_alineado"].corr(
        detalle_p["d1"]
    )
    spearman = detalle_p["PC1_alineado"].rank().corr(
        detalle_p["d1"].rank()
    )

    resumen = pd.DataFrame(
        [
            {
                "n_comunes_clustering": len(comunes_c),
                "ari_clustering_bcall": ari,
                "n_discrepancias_alineadas": int(
                    (~detalle_c["coincide"]).sum()
                ),
                "alineacion_clusters_ambigua": ambiguo,
                "n_comunes_pca": len(comunes_p),
                "pearson_pc1_original_d1": r_original,
                "signo_alineacion_pc1": signo,
                "pearson_pc1_alineado_d1": pearson,
                "spearman_pc1_alineado_d1": spearman,
                "referencia_bcall": (
                    "resultados_existentes_sin_reestimar"
                ),
                "estado": (
                    "DESCRIPTIVO_SOBRE_VOTOS_REGISTRADOS"
                ),
            }
        ]
    )

    ids = b.index.union(
        c.index, sort=False
    ).union(
        p.index, sort=False
    )

    universos = pd.DataFrame(
        {
            "diputado_id": ids,
            "grupo_bcall_disponible": ids.isin(b_grupos.index),
            "d1_disponible": ids.isin(b_validos),
            "cluster_disponible": ids.isin(c_grupos.index),
            "pc1_disponible": ids.isin(p_validos),
            "comparado_clustering": ids.isin(comunes_c),
            "comparado_pca": ids.isin(comunes_p),
        }
    )

    return {
        "resumen_comparacion.csv": resumen,
        "comparacion_clustering_bcall.csv": (
            detalle_c.rename_axis("diputado_id").reset_index()
        ),
        "correspondencia_clusters.csv": (
            correspondencia.reset_index()
        ),
        "comparacion_pca_bcall.csv": (
            detalle_p.rename_axis("diputado_id").reset_index()
        ),
        "universos_comparacion.csv": universos,
    }


def graficar_pca(tabla, ruta):
    import matplotlib.pyplot as plt

    # Agrupar posiciones coincidentes para mostrar perfiles repetidos.
    # Las correlaciones se calculan previamente sin redondear.
    puntos = tabla.assign(
        x=tabla["PC1_alineado"].round(12),
        y=tabla["d1"].round(12),
        grupo=tabla["grupo_bcall"].fillna("Sin grupo"),
    ).groupby(
        ["x", "y", "grupo"],
        as_index=False,
    ).size()

    colores = {
        "L": "#d62728",
        "R": "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for grupo, datos in puntos.groupby("grupo"):
        ax.scatter(
            datos["x"],
            datos["y"],
            s=25 + 20 * np.sqrt(datos["size"]),
            color=colores.get(grupo, "#777777"),
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            label=f"B-Call {grupo}",
        )

    ax.set(
        xlabel="PC1 alineado con d1",
        ylabel="d1 de B-Call",
        title=(
            f"PCA y B-Call · {len(tabla)} diputados comunes"
        ),
    )
    ax.grid(alpha=0.2)
    ax.legend()

    fig.text(
        0.5,
        0.01,
        "El tamaño del punto representa perfiles coincidentes.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--bcall",
        default=(
            "F4/data/results/individual/bcall/"
            "bcall_diputados.csv"
        ),
    )
    parser.add_argument(
        "--clusters",
        default="F4/data/results/grupos/clusters_diputados.csv",
    )
    parser.add_argument(
        "--pca",
        default=(
            "F4/data/results/individual/pca/"
            "coordenadas_diputados.csv"
        ),
    )
    parser.add_argument(
        "--salida",
        default="F4/data/results/comparacion",
    )
    parser.add_argument(
        "--figuras",
        default="F4/figures/comparacion",
    )

    args = parser.parse_args()
    raiz = Path(__file__).resolve().parents[3]

    def resolver(ruta):
        ruta = Path(ruta)
        return ruta if ruta.is_absolute() else raiz / ruta

    try:
        tablas = comparar(
            pd.read_csv(resolver(args.bcall)),
            pd.read_csv(resolver(args.clusters)),
            pd.read_csv(resolver(args.pca)),
        )

        salida = resolver(args.salida)
        figuras = resolver(args.figuras)

        salida.mkdir(parents=True, exist_ok=True)
        figuras.mkdir(parents=True, exist_ok=True)

        for nombre, tabla in tablas.items():
            tabla.to_csv(salida / nombre, index=False)

        graficar_pca(
            tablas["comparacion_pca_bcall.csv"],
            figuras / "pc1_vs_d1.png",
        )

    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    r = tablas["resumen_comparacion.csv"].iloc[0]

    print(
        f"Clustering: {r.n_comunes_clustering} diputados; "
        f"ARI = {r.ari_clustering_bcall:.6f}"
    )
    print(
        "Discrepancias tras alinear grupos: "
        f"{r.n_discrepancias_alineadas}"
    )
    print(f"PCA: {r.n_comunes_pca} diputados comunes")
    print(f"Signo aplicado a PC1: {r.signo_alineacion_pc1:+d}")
    print(
        "Pearson PC1 alineado / d1: "
        f"{r.pearson_pc1_alineado_d1:.6f}"
    )
    print(
        "Spearman PC1 alineado / d1: "
        f"{r.spearman_pc1_alineado_d1:.6f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())