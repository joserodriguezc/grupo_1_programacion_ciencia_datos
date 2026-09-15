def test_clave_votacion_diputado_es_unica(detalle_procesado):
    duplicados = detalle_procesado.duplicated(
        subset=["votacion_id", "diputado_id"],
        keep=False,
    )
    assert not duplicados.any(), (
        "detalle_votaciones_procesado.csv contiene claves "
        "(votacion_id, diputado_id) repetidas."
    )


def test_nombre2_no_existe(detalle_procesado):
    assert "nombre2" not in detalle_procesado.columns, (
        "Regresión detectada: nombre2 no debe formar parte del detalle procesado."
    )


def test_columnas_criticas_no_tienen_nulos(detalle_procesado):
    columnas = ["diputado_id", "descripcion"]
    nulos = detalle_procesado[columnas].isna().sum()
    assert int(nulos.sum()) == 0, (
        f"Columnas críticas con nulos: {nulos[nulos > 0].to_dict()}"
    )


def test_cardinalidad_positiva(detalle_procesado):
    assert len(detalle_procesado) > 0
