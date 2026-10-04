import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from F4.src.analisis import visualizacion as vis  # noqa: E402

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_colores_son_hexadecimales_validos() -> None:
    for paleta in (vis.GRUPOS, vis.VOTOS, vis.ESTADOS, vis.CLUSTERS):
        assert all(HEX.match(color) for color in paleta.values())


def test_cada_color_tiene_un_solo_significado() -> None:
    # Votos, grupos y clusters no comparten colores entre sí.
    usados = [*vis.GRUPOS.values(), *vis.VOTOS.values(), *vis.CLUSTERS.values()]
    assert len({c.lower() for c in usados}) == len(usados)


def test_votos_ternarios_se_asignan_en_orden() -> None:
    norma = vis.norma_ternaria()
    colores = vis.VOTO_TERNARIO(norma(np.array([-1, 0, 1])))
    esperados = [matplotlib.colors.to_rgba(vis.VOTOS[v]) for v in ("No", "Abstención", "Sí")]
    np.testing.assert_allclose(colores, esperados)


def test_norma_simetrica_centrada_en_cero() -> None:
    norma = vis.norma_simetrica([-0.2, np.nan, 0.8])

    assert (norma.vmin, norma.vcenter, norma.vmax) == (-0.8, 0.0, 0.8)
    assert norma(0.0) == pytest.approx(0.5)
    assert vis.norma_simetrica([0, 0]).vmax == 1.0


def test_coma_decimal() -> None:
    assert vis.coma(0.8569, 3) == "0,857"


def test_guardar_escribe_png_y_registra(tmp_path) -> None:
    vis.aplicar_estilo()
    fig, ax = plt.subplots()
    vis.titulo(ax, "Título", "subtítulo")
    vis.nota(fig, "Fuente: prueba.")
    registro = []

    ruta = vis.guardar(fig, tmp_path / "sub" / "figura.png", registro)
    plt.close(fig)

    assert ruta.is_file() and ruta.stat().st_size > 0
    assert registro == ["figura.png"]
    assert plt.rcParams["axes.titlelocation"] == "left"
