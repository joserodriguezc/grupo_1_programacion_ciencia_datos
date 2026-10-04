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
