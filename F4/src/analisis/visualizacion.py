"""Convenciones visuales comunes de los notebooks de F4.

Cada color tiene un único significado en todo el proyecto:

- GRUPOS: lado del eje B-Call (L rojo, R azul) y categorías externas asociadas.
- VOTOS: decisión nominal, en paleta divergente (Sí verde, No morado, Abstención gris).
- ESTADOS: resultado de un control; siempre acompañado de su etiqueta escrita.

Los índices de cohesión no se codifican por color: cada panel los identifica por su título.
Las paletas categóricas se validaron para daltonismo con la skill dataviz
(scripts/validate_palette.js); el gris de independientes es neutro a propósito.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------- tokens
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRILLA = "#e4e3df"
SUPERFICIE = "#ffffff"
SIN_DATO = "#d9d8d4"
NEUTRO = "#7a7974"  # barras o puntos de una sola serie sin significado de grupo
RAYADO = "#b8b7b2"  # borde del rayado en celdas calculadas pero no publicables
DPI = 160

# ---------------------------------------------------------------------------- paletas
GRUPOS = {
    "L": "#E74C3C",
    "R": "#3498DB",
    "Sin grupo externo": "#5E5E5E",
    "L/R distintos en el período": "#D4A017",
}
MARCADORES_GRUPO = {"L": "o", "R": "s", "Sin grupo externo": "X",
                    "L/R distintos en el período": "D"}
VOTOS = {"Sí": "#1a9850", "No": "#762a83", "Abstención": "#a3a3a3"}
VOTO_SIN_REGISTRO = SIN_DATO
ESTADOS = {"ok": "#2e7d32", "advertencia": "#c98500", "error": "#c62828"}
# Variante para texto en tablas: el ámbar se oscurece para mantener contraste sobre blanco.
ESTADOS_TEXTO = {"ok": "#2e7d32", "advertencia": "#a35a00", "error": "#c62828"}
# Agrupaciones neutrales (clusters 1..k): no implican lado del eje ni ideología.
CLUSTERS = {1: "#3d3c39", 2: "#b5b3ad"}

# ----------------------------------------------------------------------- escalas
SECUENCIAL = "Oranges"
DIVERGENTE_LR = LinearSegmentedColormap.from_list(
    "divergente_lr", [GRUPOS["L"], "#f2f1ee", GRUPOS["R"]]).with_extremes(bad=SIN_DATO)
VOTO_TERNARIO = ListedColormap([VOTOS["No"], VOTOS["Abstención"], VOTOS["Sí"]],
                               name="voto_ternario").with_extremes(bad=VOTO_SIN_REGISTRO)


def norma_ternaria() -> BoundaryNorm:
    """Asigna −1, 0 y +1 a No, Abstención y Sí en VOTO_TERNARIO."""
    return BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)


def norma_simetrica(valores) -> TwoSlopeNorm:
    """Norma centrada en 0 con límites simétricos para DIVERGENTE_LR."""
    arreglo = np.asarray(valores, dtype=float)
    maximo = float(np.nanmax(np.abs(arreglo))) if np.isfinite(arreglo).any() else 0.0
    maximo = maximo or 1.0
    return TwoSlopeNorm(vmin=-maximo, vcenter=0.0, vmax=maximo)


# ------------------------------------------------------------------------- estilo
def aplicar_estilo() -> None:
    """Parámetros de matplotlib comunes a todos los notebooks."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "font.size": 9,
        "text.color": TEXTO,
        "axes.labelcolor": TEXTO,
        "axes.edgecolor": TEXTO_SECUNDARIO,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.titlesize": 10.5,
        "axes.grid": False,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SECUNDARIO,
        "ytick.color": TEXTO_SECUNDARIO,
        "xtick.labelcolor": TEXTO,
        "ytick.labelcolor": TEXTO,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "image.cmap": SECUENCIAL,
        "savefig.dpi": DPI,
    })


def guardar(fig, ruta: str | Path, registro: list | None = None) -> Path:
    """Guarda a 160 dpi, recortado y sin metadatos variables (PNG reproducible)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", metadata={"Software": None})
    if registro is not None:
        registro.append(ruta.name)
    return ruta


def titulo(ax, texto: str, subtitulo: str | None = None) -> None:
    """Título en negrita alineado a la izquierda y subtítulo opcional en gris."""
    ax.set_title(texto, loc="left", fontweight="bold", pad=20 if subtitulo else 6)
    if subtitulo:
        ax.text(0, 1.015, subtitulo, transform=ax.transAxes, fontsize=8,
                color=TEXTO_SECUNDARIO, va="bottom", ha="left")


def nota(fig, texto: str, y: float = -0.02) -> None:
    """Nota de fuente o lectura bajo la figura, en tinta secundaria."""
    fig.text(0.01, y, texto, fontsize=7.5, color=TEXTO_SECUNDARIO, va="top", ha="left",
             wrap=True)


def coma(valor: float, decimales: int = 2) -> str:
    """Número con coma decimal y signo menos tipográfico, como en los ejes."""
    return f"{valor:.{decimales}f}".replace(".", ",").replace("-", "−")


def formato_coma(ax, eje: str = "x", decimales: int | None = None) -> None:
    """Etiquetas del eje con coma decimal; sin decimales sobrantes si decimales es None."""
    def formatear(valor, _):
        if decimales is None:
            texto = f"{valor:g}"
        else:
            texto = f"{valor:.{decimales}f}"
        return texto.replace(".", ",").replace("-", "−")

    formateador = FuncFormatter(formatear)
    (ax.xaxis if eje == "x" else ax.yaxis).set_major_formatter(formateador)


def leyenda_inferior(ax, handles: Iterable | None = None, ncol: int = 3,
                     y: float = -0.14, **kwargs):
    """Leyenda centrada bajo el eje, sin marco."""
    opciones = {"loc": "upper center", "bbox_to_anchor": (0.5, y), "ncol": ncol,
                "frameon": False}
    opciones.update(kwargs)
    if handles is None:
        return ax.legend(**opciones)
    return ax.legend(handles=list(handles), **opciones)


def parche(color: str, etiqueta: str, **kwargs) -> Patch:
    return Patch(facecolor=color, edgecolor=kwargs.pop("edgecolor", color), label=etiqueta,
                 **kwargs)


def marcador(color: str, etiqueta: str, forma: str = "o", tamano: float = 7) -> Line2D:
    return Line2D([], [], linestyle="", marker=forma, markersize=tamano,
                  markerfacecolor=color, markeredgecolor="white", label=etiqueta)


def leyenda_votos(incluir_sin_registro: bool = True) -> list[Patch]:
    """Elementos de leyenda para la vista ternaria o nominal del voto."""
    elementos = [parche(VOTOS["Sí"], "Sí (+1)"), parche(VOTOS["No"], "No (−1)"),
                 parche(VOTOS["Abstención"], "Abstención (0)")]
    if incluir_sin_registro:
        elementos.append(parche(VOTO_SIN_REGISTRO, "sin registro", edgecolor=RAYADO))
    return elementos


# ------------------------------------------------------------------ plano B-Call
# Grupo → color, forma y rótulo corto. La forma duplica el color (impresión en B/N).
ESTILO_GRUPOS = {
    grupo: {"color": GRUPOS[grupo], "marker": MARCADORES_GRUPO[grupo], "corto": corto}
    for grupo, corto in (("L", "L"), ("R", "R"), ("Sin grupo externo", "Ind."),
                         ("L/R distintos en el período", "L/R"))
}
AREA_POR_DIPUTADO = 36  # puntos² por diputado; el área es proporcional al conteo
MINIMO_ROTULO = 2       # se rotulan posiciones compartidas por al menos 2 diputados
MINIMO_MARGINAL = 5     # menos diputados no sostienen una densidad kernel
# Ancho de banda fijo (fracción del eje), igual para todos los grupos: con coordenadas
# repetidas, un ancho proporcional a la dispersión produce picos que aplastan al resto.
BANDA_MARGINAL = 0.035


def _densidad(valores, malla, banda):
    """KDE gaussiana con ancho de banda fijo; integra 1 dentro de cada grupo."""
    z = (malla[:, None] - np.asarray(valores, dtype=float)[None, :]) / banda
    return np.exp(-0.5 * z ** 2).mean(axis=1) / (banda * np.sqrt(2 * np.pi))


def _rotular_conteos(ax, rotulos):
    """Coloca cada conteo junto al borde de su burbuja y evita superponer rótulos."""
    escala = ax.figure.dpi / 72
    borde_derecho = ax.get_window_extent().x1
    colocados = []  # (centro x, base y, ancho) de cada rótulo, en píxeles
    for x, y, texto, radio in sorted(rotulos, key=lambda r: (r[1], r[0])):
        px, py = ax.transData.transform((x, y))
        separacion = radio * 0.72 + 2
        ancho, alto = 5.2 * len(texto), 10.5
        # A la derecha de la burbuja salvo que el texto se salga del eje.
        a_la_derecha = px + (separacion + ancho) * escala <= borde_derecho
        dx = separacion if a_la_derecha else -separacion
        centro = px + (dx + (ancho / 2 if a_la_derecha else -ancho / 2)) * escala
        dy = separacion
        while any(abs(centro - cx) < (ancho + cw) / 2 * escala
                  and abs(py + dy * escala - cy) < alto * escala
                  for cx, cy, cw in colocados):
            dy += alto
        colocados.append((centro, py + dy * escala, ancho))
        desplazado = dy > separacion
        ax.annotate(
            texto, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
            color=TEXTO, ha="left" if a_la_derecha else "right", va="bottom", zorder=12,
            arrowprops={"arrowstyle": "-", "color": NEUTRO, "lw": 0.6} if desplazado else None,
        )


def plano_bcall(diputados, columna_grupo, pivote, titulo, nombre_leyenda,
                etiquetas=None, limites=None):
    """Plano d1/d2 sin jitter ni redondeo: una burbuja por coordenada exacta y grupo.

    diputados: tabla indexada por diputado con d1, d2 y la columna de grupo.
    etiquetas: nombre visible de cada grupo en la leyenda (por defecto, la clave).
    Devuelve (figura, eje principal). El conteo total y las posiciones distintas van en el
    subtítulo para que la concentración sea visible.
    """
    etiquetas = etiquetas or {}
    puntos = diputados.dropna(subset=["d1", "d2"]).copy()
    puntos["grupo"] = puntos[columna_grupo].astype("object")
    grupos = [g for g in ESTILO_GRUPOS if puntos.grupo.eq(g).any()]
    agregados = (puntos.groupby(["grupo", "d1", "d2"], as_index=False).size()
                 .rename(columns={"size": "n"}))
    assert int(agregados.n.sum()) == len(puntos)

    fig = plt.figure(figsize=(11.5, 7))
    rejilla = fig.add_gridspec(2, 3, width_ratios=(6, 1.1, 2.6), height_ratios=(1.1, 5),
                               wspace=0.04, hspace=0.04,
                               left=0.06, right=0.99, top=0.88, bottom=0.12)
    ax = fig.add_subplot(rejilla[1, 0])
    ax_x = fig.add_subplot(rejilla[0, 0], sharex=ax)
    ax_y = fig.add_subplot(rejilla[1, 1], sharey=ax)

    # Burbujas grandes primero para que las pequeñas queden visibles encima.
    for _, fila in agregados.sort_values("n", ascending=False).iterrows():
        estilo = ESTILO_GRUPOS[fila.grupo]
        ax.scatter(fila.d1, fila.d2, s=AREA_POR_DIPUTADO * fila.n, color=estilo["color"],
                   marker=estilo["marker"], alpha=0.6, edgecolors="white", linewidths=1.0,
                   zorder=3)

    ax.axvline(0, color=NEUTRO, linewidth=0.8, linestyle="--", zorder=1)
    ax.grid(True, color=GRILLA, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("d1 · Posición relativa", color=TEXTO)
    ax.set_ylabel("d2 · Variabilidad", color=TEXTO)
    if limites is None:
        rango_x = puntos.d1.max() - puntos.d1.min()
        rango_y = puntos.d2.max() - puntos.d2.min()
        limites = ((puntos.d1.min() - 0.08 * rango_x, puntos.d1.max() + 0.12 * rango_x),
                   (puntos.d2.min() - 0.08 * rango_y, puntos.d2.max() + 0.12 * rango_y))
    ax.set_xlim(limites[0])
    ax.set_ylim(limites[1])
    formato_coma(ax, "x")
    formato_coma(ax, "y")

    # Marginales: una densidad por grupo con el mismo ancho de banda para todos.
    malla_x = np.linspace(*limites[0], 400)
    malla_y = np.linspace(*limites[1], 400)
    banda_x = BANDA_MARGINAL * (limites[0][1] - limites[0][0])
    banda_y = BANDA_MARGINAL * (limites[1][1] - limites[1][0])
    for grupo in grupos:
        datos = puntos.loc[puntos.grupo.eq(grupo)]
        if len(datos) < MINIMO_MARGINAL:
            continue
        color = ESTILO_GRUPOS[grupo]["color"]
        densidad_x = _densidad(datos.d1, malla_x, banda_x)
        densidad_y = _densidad(datos.d2, malla_y, banda_y)
        ax_x.fill_between(malla_x, densidad_x, color=color, alpha=0.3, linewidth=0)
        ax_x.plot(malla_x, densidad_x, color=color, linewidth=1.2)
        ax_y.fill_betweenx(malla_y, densidad_y, color=color, alpha=0.3, linewidth=0)
        ax_y.plot(densidad_y, malla_y, color=color, linewidth=1.2)
    ax_x.set_ylim(bottom=0)
    ax_y.set_xlim(left=0)
    for marginal in (ax_x, ax_y):
        marginal.set_axis_off()

    # Un rótulo por coordenada; si comparten posición varios grupos se desglosa.
    rotulos = []
    for (d1, d2), filas in agregados.groupby(["d1", "d2"]):
        total = int(filas.n.sum())
        if total < MINIMO_ROTULO:
            continue
        texto = f"×{total}"
        if len(filas) > 1:
            texto += " (" + " · ".join(
                f"{int(f.n)} {ESTILO_GRUPOS[f.grupo]['corto']}" for f in filas.itertuples()
            ) + ")"
        rotulos.append((d1, d2, texto, (AREA_POR_DIPUTADO * filas.n.max()) ** 0.5 / 2))
    _rotular_conteos(ax, rotulos)

    # El pivote va encima de todo, con borde blanco para separarlo de su burbuja.
    p = diputados.loc[pivote]
    pivote_visible = bool(np.isfinite(p.d1) and np.isfinite(p.d2))
    if pivote_visible:
        ax.scatter([p.d1], [p.d2], marker="*", s=350, color="black", edgecolors="white",
                   linewidths=1.5, zorder=10)

    n_grupo = puntos.grupo.value_counts()
    manejadores = []
    for g in grupos:
        etiqueta = f"{etiquetas.get(g, g)} (n={n_grupo[g]})"
        if g not in ("L", "R"):
            etiqueta += f" · {ESTILO_GRUPOS[g]['corto']}"
        manejadores.append(Line2D(
            [], [], linestyle="", marker=ESTILO_GRUPOS[g]["marker"], markersize=9,
            markerfacecolor=ESTILO_GRUPOS[g]["color"], markeredgecolor="white", alpha=0.8,
            label=etiqueta,
        ))
    if pivote_visible:
        manejadores.append(Line2D([], [], linestyle="", marker="*", markersize=14,
                                  markerfacecolor="black", markeredgecolor="white",
                                  label=f"Pivote {pivote}"))
    ax_leyenda = fig.add_subplot(rejilla[:, 2])
    ax_leyenda.set_axis_off()
    leyenda = ax_leyenda.legend(handles=manejadores, title=nombre_leyenda, loc="upper left",
                                frameon=False, fontsize=9, title_fontsize=9.5,
                                alignment="left")
    ax_leyenda.add_artist(leyenda)
    maximo = int(agregados.n.max())
    referencias = (1, 10, 50) if maximo >= 40 else (1, 5, 10)
    tamanos = [ax_leyenda.scatter([], [], s=AREA_POR_DIPUTADO * c, facecolors="none",
                                  edgecolors=TEXTO_SECUNDARIO,
                                  label=f"{c} diputado{'s' if c != 1 else ''}")
               for c in referencias]
    ax_leyenda.legend(handles=tamanos, title="Área = diputados\nen la misma posición",
                      loc="upper left", bbox_to_anchor=(0, 0.52), frameon=False,
                      fontsize=9, title_fontsize=9.5, labelspacing=2.6, borderpad=1.4,
                      handletextpad=2.2,
                      alignment="left")

    fig.text(0.06, 0.975, titulo, fontsize=13, fontweight="bold", color=TEXTO, va="top")
    fig.text(0.06, 0.935,
             f"{len(puntos)} diputados con d1 y d2 estimables · "
             f"{agregados[['d1', 'd2']].drop_duplicates().shape[0]} posiciones distintas",
             fontsize=10, color=TEXTO_SECUNDARIO, va="top")
    fig.text(0.06, 0.02,
             "Coordenadas originales, sin jitter ni redondeo. Se rotulan las posiciones con "
             f"{MINIMO_ROTULO} o más diputados. Marginales: densidad kernel por grupo con ancho "
             f"de banda fijo ({BANDA_MARGINAL:.1%} del eje), solo grupos con {MINIMO_MARGINAL} "
             "o más diputados.",
             fontsize=8.5, color=TEXTO_SECUNDARIO, va="bottom", wrap=True)
    return fig, ax
