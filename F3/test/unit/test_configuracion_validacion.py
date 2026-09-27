"""Contratos de entrada sobre archivos históricos y errores de dato."""

from pathlib import Path

import pandas as pd

from F3.src.nucleo.configuracion_validacion import construir_validador_interim

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


def test_cinco_archivos_historicos_y_diagnostico_militancias() -> None:
    validador = construir_validador_interim()

    for tabla, ruta in RUTAS.items():
        df = pd.read_csv(ruta, dtype="string", encoding="utf-8-sig")
        reporte = validador.validar(tabla, df)

        assert reporte.aprobado, (tabla, reporte.errores)

        if tabla == "militancias":
            assert [r.regla for r in reporte.advertencias] == [
                    "intervalos_validos",
                    "solapamientos_vigencia",
                    "correspondencia.partido_id.partido_nombre",
            ]
            assert reporte.advertencias[0].filas == (380,)
            assert "varios códigos" in reporte.advertencias[2].detalle

            

def test_fecha_obligatoria_y_total_invalido_bloquean() -> None:
    df = pd.read_csv(
        RUTAS["proyecto_ley"],
        dtype="string",
        encoding="utf-8-sig",
    )
    df.loc[0, "Fecha"] = ""
    df.loc[1, "TotalSi"] = "-1"

    reporte = construir_validador_interim().validar("proyecto_ley", df)

    assert not reporte.aprobado
    assert {r.regla for r in reporte.errores} == {
        "completitud",
        "fechas_validas",
        "enteros_no_negativos",
    }


def test_militancia_observada_no_cambia_durante_validacion() -> None:
    df = pd.read_csv(
        RUTAS["militancias"],
        dtype="string",
        encoding="utf-8-sig",
    )
    original = df.copy(deep=True)

    construir_validador_interim().validar("militancias", df)

    pd.testing.assert_frame_equal(df, original)
    
def test_codigo_con_dos_descripciones_se_informa() -> None:
    df = pd.read_csv(
        RUTAS["detalle_votaciones"],
        dtype="string",
        encoding="utf-8-sig",
    )
    df.loc[0, "opcion_codigo"] = df.loc[1, "opcion_codigo"]
    df.loc[0, "opcion_voto"] = "Otra descripción"

    reporte = construir_validador_interim().validar(
        "detalle_votaciones", df
    )

    assert any(
        registro.regla == "correspondencia.opcion_codigo.opcion_voto"
        for registro in reporte.advertencias
    )