def test_cardinalidad_preservada(
    big_table_analitica,
    detalle_procesado,
):
    assert len(big_table_analitica) == len(
        detalle_procesado
    )

def test_clave_unica(
    big_table_analitica,
):
    duplicados = big_table_analitica.duplicated(
        subset=[
            "votacion_id",
            "diputado_id",
        ],
        keep=False,
    )

    assert not duplicados.any()

def test_partido_alias_no_nulo(
    big_table_analitica,
):
    faltantes = (
        big_table_analitica[
            "partido_alias"
        ]
        .isna()
    )

    assert not faltantes.any()