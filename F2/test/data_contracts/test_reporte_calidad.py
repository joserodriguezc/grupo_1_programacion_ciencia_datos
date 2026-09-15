COLUMNAS_REQUERIDAS = {
    "tabla",
    "diputado_id",
    "tipo",
    "descripcion",
    "regla_aplicada",
}


def test_reporte_tiene_columnas_requeridas(
    reporte_calidad,
):
    faltantes = (
        COLUMNAS_REQUERIDAS
        - set(reporte_calidad.columns)
    )

    assert not faltantes, (
        "Faltan columnas en reporte_calidad.csv: "
        f"{sorted(faltantes)}"
    )


def test_campos_descriptivos_no_estan_vacios(
    reporte_calidad,
):
    for columna in [
        "tabla",
        "descripcion",
        "regla_aplicada",
    ]:
        serie = reporte_calidad[columna]

        assert serie.notna().all(), (
            f"{columna} contiene nulos."
        )

        assert (
            serie.astype(str)
            .str.strip()
            .ne("")
            .all()
        ), (
            f"{columna} contiene cadenas vacías."
        )


def test_excepciones_particulares_permanecen(
    reporte_calidad,
):
    excepciones = reporte_calidad[
        reporte_calidad["tipo"]
        == "excepcion_particular"
    ]

    ids = set(
        excepciones["diputado_id"]
        .astype(int)
    )

    esperados = {
        1017,  # Carter
        1114,  # Bugueño
        1180,  # Veloso
    }

    assert esperados.issubset(ids), (
        "Faltan excepciones documentadas. "
        f"Esperadas: {esperados}; "
        f"observadas: {ids}"
    )