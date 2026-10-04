"""Contrato de datos (A04): el repositorio lo cumple y los módulos usan sus rutas."""

from pathlib import Path

import pandas as pd
import pytest

from F4.src.analisis import (
    afinidad,
    cobertura,
    codificacion,
    cohesion,
    contratos,
    corte,
    matrices,
    posicion_partidos,
    sensibilidad,
)

RAIZ = Path(__file__).resolve().parents[3]


def test_repositorio_cumple_el_contrato() -> None:
    # El manifiesto de entrega se genera al final, con exportacion.py.
    assert contratos.validar_repositorio(RAIZ, omitir=("manifiesto_entrega",)) == []


@pytest.mark.parametrize(("ruta_modulo", "nombre"), [
    (codificacion.RUTA_SALIDA_POR_DEFECTO, "votos"),
    (matrices.RUTA_ENTRADA_POR_DEFECTO, "votos"),
    (matrices.RUTA_AUDITORIA_POR_DEFECTO, "auditoria"),
    (corte.RUTA_MANIFIESTO_POR_DEFECTO, "manifiesto_corte"),
    (posicion_partidos.RUTA_MATRIZ_POR_DEFECTO, "ternaria"),
    (posicion_partidos.RUTA_AFILIACION_POR_DEFECTO, "afiliacion"),
    (posicion_partidos.RUTA_SALIDA_POR_DEFECTO, "posicion"),
    (afinidad.RUTA_MATRIZ_POR_DEFECTO, "nominal"),
    (afinidad.RUTA_SALIDA_POR_DEFECTO, "afinidad"),
    (afinidad.RUTA_MATRIZ_HAMMING_POR_DEFECTO, "matriz_hamming"),
    (afinidad.RUTA_HAMMING_PARTIDOS_POR_DEFECTO, "hamming_partidos"),
    (sensibilidad.RUTA_MATRIZ_POR_DEFECTO, "ternaria"),
    (sensibilidad.RUTA_NOMINAL_POR_DEFECTO, "nominal"),
    (sensibilidad.RUTA_SALIDA_POR_DEFECTO, "sensibilidad"),
    (cohesion.SALIDA, "cohesion"),
    (cohesion.SALIDA_RESUMEN, "resumen_cohesion"),
    (f"{cobertura.REPORTES}/{cobertura.SALIDAS['participacion']}", "participacion"),
    (f"{cobertura.REPORTES}/{cobertura.SALIDAS['partido_votacion']}", "conteos_participacion"),
])
def test_rutas_de_los_modulos_coinciden_con_el_contrato(ruta_modulo, nombre) -> None:
    assert Path(ruta_modulo).as_posix() == contratos.SALIDAS[nombre].ruta


def test_matrices_escribe_en_las_rutas_del_contrato() -> None:
    for nombre in ("nominal", "ternaria", "binaria", "mascara", "afiliacion"):
        assert Path(contratos.SALIDAS[nombre].ruta).parent == matrices.DIRECTORIO_SALIDA_POR_DEFECTO


def test_detecta_columna_faltante_clave_duplicada_y_codigo_desconocido() -> None:
    tabla = pd.DataFrame({
        "diputado_id": [1, 1], "participacion": [0.5, 0.9], "incluido": [True, False],
        "razon_exclusion": [None, "CODIGO_INVENTADO"]})
    problemas = contratos.validar_tabla("bcall_seleccion", tabla)
    assert "bcall_seleccion: clave ('diputado_id',) duplicada" in problemas
    assert any("CODIGO_INVENTADO" in p for p in problemas)

    sin_columna = tabla.drop(columns="participacion").drop_duplicates("diputado_id", keep="last")
    assert contratos.validar_tabla("bcall_seleccion", sin_columna) == [
        "bcall_seleccion: falta la columna participacion",
        "bcall_seleccion.razon_exclusion: códigos fuera del contrato ['CODIGO_INVENTADO']",
    ]


def test_codigos_en_listas_json_y_separados_por_punto_y_coma() -> None:
    tabla = pd.DataFrame({
        "votacion_id": [1, 2], "orientacion": [1, -1], "incluida_por_varianza": [True, False],
        "razones_exclusion": ['[]', '["SIN_VARIANZA_O_MENOS_DE_DOS_OBSERVACIONES", "OTRO"]']})
    assert contratos.validar_tabla("bcall_votaciones", tabla) == [
        "bcall_votaciones.razones_exclusion: códigos fuera del contrato ['OTRO']"]


def test_equivalencias_apuntan_al_catalogo_normativo() -> None:
    assert set(contratos.EQUIVALENCIAS.values()) <= contratos.CATALOGO_NORMATIVO


def test_falta_archivo(tmp_path) -> None:
    problemas = contratos.validar_repositorio(tmp_path)
    assert len(problemas) == len(contratos.SALIDAS)
    assert all("falta o está vacío" in p for p in problemas)
