"""Comprueba referencias entre las cinco entradas interim."""

from pathlib import Path

import pandas as pd

from F3.src.nucleo.validador_integracion import ValidadorIntegracion

INTERIM = Path(__file__).resolve().parents[3] / "F3" / "data" / "interim"

RUTAS = {
    "periodos": INTERIM / "periodos.csv",
    "diputados": INTERIM / "diputados.csv",
    "militancias": INTERIM / "militancias.csv",
    "proyecto_ley": (
        INTERIM / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"
    ),
    "detalle_votaciones": (
        INTERIM / "votaciones" / "detalle_votaciones.csv"
    ),
}


def cargar_interim() -> dict[str, pd.DataFrame]:
    return {
        tabla: pd.read_csv(ruta, dtype="string", encoding="utf-8-sig")
        for tabla, ruta in RUTAS.items()
    }


def test_referencias_de_f3_tienen_correspondencia() -> None:
    tablas = cargar_interim()
    originales = {
        nombre: df.copy(deep=True)
        for nombre, df in tablas.items()
    }

    reporte = ValidadorIntegracion().validar_referencias_interim(tablas)

    assert reporte.aprobado
    assert len(reporte.registros) == 6

    for nombre, df in tablas.items():
        pd.testing.assert_frame_equal(df, originales[nombre])


def test_registra_votacion_y_diputado_sin_referencia() -> None:
    tablas = cargar_interim()
    tablas["detalle_votaciones"].loc[0, "votacion_id"] = "999999"
    tablas["detalle_votaciones"].loc[1, "diputado_id"] = "999999"

    reporte = ValidadorIntegracion().validar_referencias_interim(tablas)

    assert not reporte.aprobado
    assert any(registro.filas == (0,) for registro in reporte.errores)
    assert any(registro.filas == (1,) for registro in reporte.errores)


def test_falta_de_tabla_devuelve_error_en_reporte() -> None:
    tablas = cargar_interim()
    del tablas["periodos"]

    reporte = ValidadorIntegracion().validar_referencias_interim(tablas)

    assert not reporte.aprobado
    assert "periodos" in reporte.errores[0].detalle
    
def test_consistencia_historica_sin_cambios_en_las_tablas() -> None:
    tablas = cargar_interim()
    copia = {
        nombre: df.copy(deep=True)
        for nombre, df in tablas.items()
    }

    reporte = ValidadorIntegracion().validar_consistencia_interim(tablas)

    assert reporte.aprobado
    assert len(reporte.registros) == 22

    for nombre, df in tablas.items():
        pd.testing.assert_frame_equal(df, copia[nombre])


def test_diferencia_de_identidad_se_informa_sin_sobrescribir() -> None:
    tablas = cargar_interim()
    tablas["detalle_votaciones"].loc[0, "nombre"] = "Nombre diferente"

    reporte = ValidadorIntegracion().validar_consistencia_interim(tablas)

    assert reporte.aprobado
    assert reporte.advertencias[0].regla == "identidad_diputado"
    assert 0 in reporte.advertencias[0].filas
    assert tablas["detalle_votaciones"].loc[0, "nombre"] == "Nombre diferente"


def test_metadatos_internos_y_catalogo_diferentes_bloquean() -> None:
    tablas = cargar_interim()
    votacion = tablas["detalle_votaciones"].loc[0, "votacion_id"]

    tablas["detalle_votaciones"].loc[0, "total_si"] = "999"
    tablas["proyecto_ley"].loc[
        tablas["proyecto_ley"]["Id"].eq(votacion), "Tipo"
    ] = "Otro tipo"

    reporte = ValidadorIntegracion().validar_consistencia_interim(tablas)

    assert not reporte.aprobado
    reglas = {registro.regla for registro in reporte.errores}
    assert "metadatos_constantes.total_si" in reglas
    assert "proyecto_vs_detalle.Tipo" in reglas

def test_periodos_y_totales_historicos_concuerdan() -> None:
    reporte = ValidadorIntegracion().validar_periodos_y_totales_interim(
        cargar_interim()
    )

    assert reporte.aprobado
    assert [r.regla for r in reporte.registros] == [
        "periodo_unico",
        "conciliacion_votos",
    ]


def test_fecha_exacta_de_termino_pertenece_al_periodo() -> None:
    tablas = cargar_interim()
    periodo = tablas["periodos"].loc[
        tablas["periodos"]["periodo_id"].eq("10")
    ].iloc[0]

    tablas["proyecto_ley"].loc[0, "Fecha"] = periodo["fecha_termino"]

    reporte = ValidadorIntegracion().validar_periodos_y_totales_interim(
        tablas
    )
    assert reporte.aprobado


def test_periodos_ambiguos_y_total_oficial_incorrecto_bloquean() -> None:
    tablas = cargar_interim()
    copia = tablas["periodos"].loc[
        tablas["periodos"]["periodo_id"].eq("10")
    ].copy()
    copia.loc[:, "periodo_id"] = "999"

    tablas["periodos"] = pd.concat(
        [tablas["periodos"], copia],
        ignore_index=True,
    )
    tablas["proyecto_ley"].loc[0, "TotalSi"] = "999"

    reporte = ValidadorIntegracion().validar_periodos_y_totales_interim(
        tablas
    )

    assert not reporte.aprobado
    assert {r.regla for r in reporte.errores} == {
        "periodo_unico",
        "conciliacion_votos",
    }