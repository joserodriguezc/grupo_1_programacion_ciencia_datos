"""Asignación temporal de militancias a votos."""

import pandas as pd

COLUMNAS_DIAGNOSTICO = [
    "fila_voto_id",
    "diputado_id",
    "fecha",
    "militancias_vigentes",
    "estado",
]

COLUMNAS_ASIGNACION = [
    "fila_voto_id",
    "partido_id",
    "partido_nombre",
    "partido_alias",
]


def preparar_entradas(
    votos: pd.DataFrame,
    militancias: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara una vez las mismas entradas para ambos asignadores.

    Crea fila_voto_id si aún no existe y convierte identificadores y fechas.
    No modifica los DataFrame recibidos.
    """
    columnas_votos = {"diputado_id", "fecha"}
    if "fila_voto_id" not in votos.columns:
        columnas_votos.add("votacion_id")

    columnas_militancias = {
        "diputado_id",
        "partido_id",
        "partido_nombre",
        "partido_alias",
        "fecha_inicio",
        "fecha_termino",
    }

    faltantes_votos = columnas_votos - set(votos.columns)
    faltantes_militancias = columnas_militancias - set(militancias.columns)

    if faltantes_votos or faltantes_militancias:
        raise ValueError(
            f"Faltan columnas en votos: {sorted(faltantes_votos)}; "
            f"en militancias: {sorted(faltantes_militancias)}"
        )

    votos_preparados = votos.copy()
    militancias_preparadas = militancias.copy()

    votos_preparados["diputado_id"] = (
        votos_preparados["diputado_id"].astype("string")
    )

    if "fila_voto_id" not in votos_preparados.columns:
        votos_preparados["votacion_id"] = (
            votos_preparados["votacion_id"].astype("string")
        )

        if votos_preparados["votacion_id"].isna().any():
            raise ValueError("Hay votos sin votacion_id")

        votos_preparados["fila_voto_id"] = (
            votos_preparados["votacion_id"]
            + "_"
            + votos_preparados["diputado_id"]
        )
    else:
        votos_preparados["fila_voto_id"] = (
            votos_preparados["fila_voto_id"].astype("string")
        )

    votos_preparados["fecha"] = pd.to_datetime(
        votos_preparados["fecha"],
        errors="raise",
    )

    militancias_preparadas["diputado_id"] = (
        militancias_preparadas["diputado_id"].astype("string")
    )

    for columna in ("partido_id", "partido_nombre", "partido_alias"):
        militancias_preparadas[columna] = (
            militancias_preparadas[columna].astype("string")
        )

    militancias_preparadas["fecha_inicio"] = pd.to_datetime(
        militancias_preparadas["fecha_inicio"],
        errors="raise",
    )
    militancias_preparadas["fecha_termino"] = pd.to_datetime(
        militancias_preparadas["fecha_termino"],
        errors="raise",
    )

    if votos_preparados[
        ["fila_voto_id", "diputado_id", "fecha"]
    ].isna().any().any():
        raise ValueError("Hay votos sin identificador, diputado o fecha")

    if votos_preparados["fila_voto_id"].duplicated().any():
        raise ValueError("fila_voto_id debe ser único por voto")

    if militancias_preparadas[
        [
            "diputado_id",
            "partido_id",
            "partido_nombre",
            "partido_alias",
            "fecha_inicio",
        ]
    ].isna().any().any():
        raise ValueError(
            "Hay militancias sin diputado, partido o fecha_inicio"
        )

    return votos_preparados, militancias_preparadas


def asignar_iterativo(
    big_table: pd.DataFrame,
    df_militancias: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrae la asignación iterativa de F2_04.

    Recibe entradas ya preparadas. Devuelve
    (diagnosticos, asignaciones), sin modificar big_table.
    """
    militancias_por_diputado = {
        diputado_id: grupo.sort_values("fecha_inicio").reset_index(drop=True)
        for diputado_id, grupo in df_militancias.groupby("diputado_id")
    }

    diagnostico_militancia = []
    resultados_militancia = []

    for _, voto in big_table[
        ["fila_voto_id", "diputado_id", "fecha"]
    ].iterrows():
        grupo = militancias_por_diputado.get(voto["diputado_id"])
        fecha = voto["fecha"]

        if grupo is None:
            vigentes = grupo
            cantidad = 0
        else:
            vigentes = grupo[
                (grupo["fecha_inicio"] <= fecha)
                & (
                    grupo["fecha_termino"].isna()
                    | (fecha <= grupo["fecha_termino"])
                )
            ]
            cantidad = len(vigentes)

        estado = (
            "OK"
            if cantidad == 1
            else ("SIN_MILITANCIA" if cantidad == 0 else "AMBIGUA")
        )

        diagnostico_militancia.append(
            {
                "fila_voto_id": voto["fila_voto_id"],
                "diputado_id": voto["diputado_id"],
                "fecha": fecha,
                "militancias_vigentes": cantidad,
                "estado": estado,
            }
        )

        fila_militancia = vigentes.iloc[0] if cantidad == 1 else None
        resultados_militancia.append(
            {
                "fila_voto_id": voto["fila_voto_id"],
                "partido_id": (
                    fila_militancia["partido_id"]
                    if fila_militancia is not None
                    else None
                ),
                "partido_nombre": (
                    fila_militancia["partido_nombre"]
                    if fila_militancia is not None
                    else None
                ),
                "partido_alias": (
                    fila_militancia["partido_alias"]
                    if fila_militancia is not None
                    else None
                ),
            }
        )

    diagnosticos = pd.DataFrame(
        diagnostico_militancia,
        columns=COLUMNAS_DIAGNOSTICO,
    )
    asignaciones = pd.DataFrame(
        resultados_militancia,
        columns=COLUMNAS_ASIGNACION,
    )

    return diagnosticos, asignaciones


def asignar_vectorizado(
    big_table: pd.DataFrame,
    df_militancias: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asigna mediante cruce por diputado y filtro temporal vectorizado.

    Recibe las mismas entradas ya preparadas que asignar_iterativo().
    """
    votos = big_table[
        ["fila_voto_id", "diputado_id", "fecha"]
    ]

    militancias = df_militancias[
        [
            "diputado_id",
            "partido_id",
            "partido_nombre",
            "partido_alias",
            "fecha_inicio",
            "fecha_termino",
        ]
    ]

    candidatos = votos.merge(
        militancias,
        on="diputado_id",
        how="left",
        sort=False,
    )

    vigente = (
        candidatos["fecha_inicio"].notna()
        & (candidatos["fecha_inicio"] <= candidatos["fecha"])
        & (
            candidatos["fecha_termino"].isna()
            | (candidatos["fecha"] <= candidatos["fecha_termino"])
        )
    )

    coincidencias = candidatos.loc[vigente]

    cantidades = coincidencias.groupby("fila_voto_id").size()

    diagnosticos = votos.copy()
    diagnosticos["militancias_vigentes"] = (
        diagnosticos["fila_voto_id"]
        .map(cantidades)
        .fillna(0)
        .astype("int64")
    )
    diagnosticos["estado"] = (
        diagnosticos["militancias_vigentes"]
        .map({0: "SIN_MILITANCIA", 1: "OK"})
        .fillna("AMBIGUA")
    )

    ids_con_asignacion_unica = diagnosticos.loc[
        diagnosticos["estado"] == "OK",
        "fila_voto_id",
    ]

    partidos_unicos = coincidencias.loc[
        coincidencias["fila_voto_id"].isin(
            ids_con_asignacion_unica
        ),
        COLUMNAS_ASIGNACION,
    ]

    asignaciones = votos[["fila_voto_id"]].merge(
        partidos_unicos,
        on="fila_voto_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    return (
        diagnosticos[COLUMNAS_DIAGNOSTICO],
        asignaciones[COLUMNAS_ASIGNACION],
    )


def exigir_asignacion_unica(diagnosticos: pd.DataFrame) -> None:
    """Detiene la integración si algún voto no tiene un partido único."""
    casos_invalidos = diagnosticos.loc[
        diagnosticos["estado"] != "OK"
    ]

    if not casos_invalidos.empty:
        raise ValueError(
            f"{len(casos_invalidos)} votos no tienen exactamente "
            "una militancia vigente. Revisar los diagnósticos."
        )