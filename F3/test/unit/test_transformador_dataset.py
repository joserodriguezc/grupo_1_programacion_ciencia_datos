"""Contrato de transformación frente a la línea base F2 y casos de fallo."""

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from F3.src.nucleo.generar_reportes_validacion import RUTAS
from F3.src.nucleo.normalizacion import normalizar_fecha
from F3.src.nucleo.transformador_dataset import TransformadorDataset

PROCESSED_F2 = Path(__file__).resolve().parents[3] / "F2" / "data" / "processed"


def cargar_tablas() -> dict[str, pd.DataFrame]:
    return {
        nombre: pd.read_csv(ruta, dtype="string", encoding="utf-8-sig")
        for nombre, ruta in RUTAS.items()
    }


def test_cinco_salidas_equivalentes_a_f2_y_entradas_intactas() -> None:
    entradas = cargar_tablas()
    originales = {nombre: df.copy(deep=True) for nombre, df in entradas.items()}

    resultado = TransformadorDataset().transformar(entradas)

    for nombre, observado in resultado.tablas.items():
        esperado = pd.read_csv(PROCESSED_F2 / f"{nombre}.csv", dtype="string")
        exportado = pd.read_csv(StringIO(observado.to_csv(index=False)), dtype="string")
        pd.testing.assert_frame_equal(exportado, esperado)

    for nombre, df in entradas.items():
        pd.testing.assert_frame_equal(df, originales[nombre])

    observadas = resultado.tablas["militancias_observadas"]
    analiticas = resultado.tablas["militancias_analiticas"]
    assert "LIBERAL" in set(observadas["partido_id"])
    assert "LIBERAL" not in set(analiticas["partido_id"])
    assert len(resultado.reporte_calidad.query("tipo == 'intervalo_invertido'")) == 1


def test_selectores_resisten_un_cambio_en_el_orden_de_militancias() -> None:
    entradas = cargar_tablas()
    entradas["militancias"] = (
        entradas["militancias"].sample(frac=1, random_state=7).reset_index(drop=True)
    )

    resultado = TransformadorDataset().transformar(entradas)
    esperado = pd.read_csv(PROCESSED_F2 / "militancias_analiticas.csv", dtype="string")
    obtenido = pd.read_csv(
        StringIO(resultado.tablas["militancias_analiticas"].to_csv(index=False)), dtype="string"
    )
    columnas = ["diputado_id", "partido_id", "fecha_inicio", "fecha_termino"]
    pd.testing.assert_frame_equal(
        obtenido.sort_values(columnas).reset_index(drop=True),
        esperado.sort_values(columnas).reset_index(drop=True),
    )


def test_excepcion_historica_que_no_coincide_detiene_proceso() -> None:
    entradas = cargar_tablas()
    militancias = entradas["militancias"]
    mascara = militancias["diputado_id"].eq("1114") & militancias["partido_id"].eq("FRVS")
    militancias.loc[mascara, "fecha_inicio"] = "2022-03-12 00:00:00"

    with pytest.raises(ValueError, match="militancia única para 1114/FRVS"):
        TransformadorDataset().transformar(entradas)


def test_fecha_obligatoria_no_admite_nulo() -> None:
    with pytest.raises(ValueError, match="fecha: fecha inválida"):
        normalizar_fecha(pd.Series([None]), "fecha", permitir_nulos=False)

    assert pd.isna(normalizar_fecha(pd.Series([None]), "fecha", permitir_nulos=True).iloc[0])
