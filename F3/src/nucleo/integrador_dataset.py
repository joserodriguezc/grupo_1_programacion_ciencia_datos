"""Construye la big table analítica a partir de las tablas procesadas."""

import pandas as pd

from .asignadores import (
    asignar_vectorizado,
    exigir_asignacion_unica,
    preparar_entradas,
)

COLUMNAS_PROYECTO = [
    "Id",
    "numero_boletin",
    "TipoVotacionProyectoLey",
    "Articulo",
    "TramiteConstitucional",
    "TramiteReglamentario",
]

COLUMNAS_DIPUTADOS = [
    "diputado_id",
    "fecha_nacimiento",
    "sexo_valor",
    "sexo_desc",
    "periodo_id",
    "fecha_inicio_periodo",
    "fecha_termino_periodo",
    "tiene_votos",
]

COLUMNAS_IDENTIDAD = [
    "nombre",
    "apellido_paterno",
    "apellido_materno",
]


def construir_big_table(
    detalle: pd.DataFrame,
    proyecto: pd.DataFrame,
    diputados: pd.DataFrame,
    militancias: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (big_table_exportable, diagnosticos_de_militancia).

    Recibe las cuatro tablas procesadas y no modifica las entradas.
    """
    votos, militancias_preparadas = preparar_entradas(
        detalle,
        militancias,
    )

    proyecto_preparado = proyecto.copy()
    diputados_preparados = diputados.copy()

    proyecto_preparado["Id"] = (
        proyecto_preparado["Id"].astype("string")
    )
    diputados_preparados["diputado_id"] = (
        diputados_preparados["diputado_id"].astype("string")
    )

    # 1. Incorporar los campos del proyecto de ley.
    big_table = votos.merge(
        proyecto_preparado[COLUMNAS_PROYECTO],
        how="left",
        left_on="votacion_id",
        right_on="Id",
        validate="many_to_one",
        sort=False,
    )

    big_table = (
        big_table
        .drop(columns="Id")
        .rename(
            columns={
                "TipoVotacionProyectoLey": (
                    "tipo_votacion_proyecto_ley"
                ),
                "Articulo": "articulo",
                "TramiteConstitucional": (
                    "tramite_constitucional"
                ),
                "TramiteReglamentario": (
                    "tramite_reglamentario"
                ),
            }
        )
    )

    columnas_proyecto_obligatorias = [
        "numero_boletin",
        "tipo_votacion_proyecto_ley",
        "tramite_constitucional",
        "tramite_reglamentario",
    ]
    # Articulo puede ser nulo en la fuente; F2_04 lo documenta.
    if big_table[
        columnas_proyecto_obligatorias
    ].isna().any(axis=1).any():
        raise ValueError(
            "Hay votos sin datos obligatorios del proyecto de ley"
        )

    # 2. Comprobar que la identidad del diputado coincide entre tablas.
    identidad = votos[
        ["fila_voto_id", "diputado_id", *COLUMNAS_IDENTIDAD]
    ].merge(
        diputados_preparados[
            ["diputado_id", *COLUMNAS_IDENTIDAD]
        ],
        on="diputado_id",
        how="left",
        suffixes=("_detalle", "_maestra"),
        validate="many_to_one",
        sort=False,
    )

    for columna in COLUMNAS_IDENTIDAD:
        coincide = (
            identidad[f"{columna}_detalle"]
            .eq(identidad[f"{columna}_maestra"])
            .fillna(False)
        )
        if not coincide.all():
            raise ValueError(
                f"Identidad de diputado inconsistente: {columna}"
            )

    # 3. Incorporar los campos nuevos del diputado.
    big_table = big_table.merge(
        diputados_preparados[COLUMNAS_DIPUTADOS],
        how="left",
        on="diputado_id",
        validate="many_to_one",
        sort=False,
    )

    columnas_diputado_obligatorias = [
        "fecha_nacimiento",
        "sexo_valor",
        "sexo_desc",
        "periodo_id",
        "fecha_inicio_periodo",
        "tiene_votos",
    ]
    # fecha_termino_periodo puede ser nula si el período sigue vigente.
    if big_table[
        columnas_diputado_obligatorias
    ].isna().any(axis=1).any():
        raise ValueError(
            "Hay votos sin datos obligatorios del diputado"
        )

    # 4. Asignar la militancia vigente por fecha de voto.
    diagnosticos, asignaciones = asignar_vectorizado(
        big_table,
        militancias_preparadas,
    )
    exigir_asignacion_unica(diagnosticos)

    big_table = big_table.merge(
        asignaciones,
        on="fila_voto_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if big_table[
        ["partido_id", "partido_nombre", "partido_alias"]
    ].isna().any().any():
        raise ValueError(
            "Hay votos sin partido después de la asignación"
        )

    # 5. Contratos finales de la integración.
    if len(big_table) != len(votos):
        raise ValueError(
            "La integración cambió la cantidad de votos"
        )

    if big_table.duplicated(
        ["votacion_id", "diputado_id"]
    ).any():
        raise ValueError(
            "La clave votacion_id + diputado_id no es única"
        )

    if big_table.columns.duplicated().any():
        raise ValueError(
            "La big table tiene columnas duplicadas"
        )

    sufijos = [
        columna
        for columna in big_table.columns
        if columna.endswith(("_x", "_y"))
    ]
    if sufijos:
        raise ValueError(
            f"Hay sufijos accidentales de merge: {sufijos}"
        )

    # F2_04 usa fila_voto_id durante la integración,
    # pero lo excluye del CSV final.
    big_table_exportable = (
        big_table
        .drop(columns="fila_voto_id")
        .sort_values(
            ["fecha", "votacion_id", "diputado_id"]
        )
        .reset_index(drop=True)
    )

    return big_table_exportable, diagnosticos