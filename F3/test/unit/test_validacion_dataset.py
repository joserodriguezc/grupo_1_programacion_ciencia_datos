import pandas as pd

from F3.src.nucleo.validacion_dataset import (
    ValidadorDataset,
    regla_intervalos_validos,
    regla_solapamientos_vigencia,
)


def test_intervalo_invertido_no_oculta_solapamiento() -> None:
    df = pd.DataFrame(
        {
            "diputado_id": ["1180", "1017", "1017"],
            "fecha_inicio": ["2026-03-11", "2022-03-11", "2023-03-11"],
            "fecha_termino": ["2026-03-10", "2024-03-11", "2025-03-11"],
        },
        index=[380, 70, 72],
    )
    copia = df.copy(deep=True)

    reporte = ValidadorDataset(
        {
            "militancias": (
                regla_intervalos_validos(estado_invertido="advertencia"),
                regla_solapamientos_vigencia(estado_solapamiento="advertencia"),
            )
        }
    ).validar("militancias", df)

    assert reporte.aprobado
    assert [r.regla for r in reporte.advertencias] == [
        "intervalos_validos",
        "solapamientos_vigencia",
    ]
    assert reporte.advertencias[0].filas == (380,)
    assert reporte.advertencias[1].filas == (70, 72)
    pd.testing.assert_frame_equal(df, copia)


def test_fecha_malformada_bloquea_validacion() -> None:
    df = pd.DataFrame(
        {
            "fecha_inicio": ["2022-03-11", "2022-03-11"],
            "fecha_termino": ["", "sin-fecha"],
        }
    )

    reporte = ValidadorDataset(
        {
            "militancias": (
                regla_intervalos_validos(estado_invertido="advertencia"),
            )
        }
    ).validar("militancias", df)

    assert not reporte.aprobado
    assert reporte.errores[0].filas == (1,)


def test_periodo_no_admite_termino_abierto() -> None:
    df = pd.DataFrame(
        {"fecha_inicio": ["2022-03-11"], "fecha_termino": [""]}
    )

    reporte = ValidadorDataset(
        {
            "periodos": (
                regla_intervalos_validos(permitir_termino_abierto=False),
            )
        }
    ).validar("periodos", df)

    assert not reporte.aprobado
    assert reporte.errores[0].filas == (0,)