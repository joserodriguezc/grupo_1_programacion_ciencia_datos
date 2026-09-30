"""C02: pivote y cotejo externo; el JSON nunca forma clusters ni cambia puntajes."""

import pandas as pd


def clasificacion_partidos(documento: dict) -> pd.Series:
    """Leer registros por nombre exacto; null y desconocidos quedan sin asignación."""
    registros = documento["registros"]
    nombres = [r["partido_nombre"] for r in registros]
    codigos = [r["codigo_binario"] for r in registros]
    if len(set(nombres)) != len(nombres):
        raise ValueError("El JSON contiene nombres de partido duplicados.")
    if any(c is not None and (type(c) is not int or c not in (0, 1)) for c in codigos):
        raise ValueError("codigo_binario debe ser 0, 1 o null.")
    return pd.Series(codigos, index=nombres, dtype="Int64", name="grupo_externo")


def elegir_pivote(
    matriz: pd.DataFrame, afiliaciones: pd.DataFrame, documento: dict,
    threshold: float = 0.1, partidos_preferidos: tuple[str, ...] = (
        "Unión Demócrata Independiente", "Partido Republicano"
    ),
):
    """Derecha externa clara, mismo partido observado, más participación y menor ID.

    Solo el pivote se selecciona con el JSON. Resto de diputados permanece intacto.
    afiliaciones debe traer diputado_id, votacion_id y partido_nombre históricos.
    IDs deben ser enteros para aplicar desempate por menor identificador.
    """
    mapa = clasificacion_partidos(documento)
    requeridas = {"diputado_id", "votacion_id", "partido_nombre"}
    if not requeridas.issubset(afiliaciones):
        raise ValueError("Faltan columnas de afiliación histórica.")
    if afiliaciones.duplicated(["diputado_id", "votacion_id"]).any():
        raise ValueError("Afiliación duplicada por diputado y votación.")
    a = afiliaciones[
        afiliaciones.diputado_id.isin(matriz.index)
        & afiliaciones.votacion_id.isin(matriz.columns)
    ]
    participacion = matriz.notna().sum(axis=1) / matriz.shape[1]
    candidatos = []
    for i, g in a.groupby("diputado_id"):
        if g.partido_nombre.isna().any() or g.partido_nombre.nunique() != 1:
            continue
        partido = g.partido_nombre.iloc[0]
        codigo = mapa.get(partido, pd.NA)
        if pd.notna(codigo) and codigo == 1 and partido in partidos_preferidos:
            if participacion.loc[i] > threshold:
                candidatos.append((i, participacion.loc[i]))
    if not candidatos:
        raise ValueError("No hay pivote elegible con afiliación de derecha consistente.")
    return sorted(candidatos, key=lambda x: (-x[1], x[0]))[0][0]


def cotejar_asignacion(
    grupos_bcall: pd.Series, afiliaciones: pd.DataFrame, documento: dict
) -> pd.DataFrame:
    """Comparar por diputado×votación sin atribuir etiqueta externa a independientes.

    grupos_bcall contiene L/R efectivos (columna grupo_bcall de calcular_auto).
    No hay denominador de concordancia hasta filtrar comparable=True.
    Una divergencia es contraste de comportamiento con covariable, no error.
    """
    if grupos_bcall.index.has_duplicates or not grupos_bcall.isin(["L", "R"]).all():
        raise ValueError("Se requieren IDs únicos y grupos B-Call L/R.")
    a = afiliaciones.copy()
    if not {"diputado_id", "votacion_id", "partido_nombre"}.issubset(a):
        raise ValueError("Faltan columnas de afiliación histórica.")
    if a.duplicated(["diputado_id", "votacion_id"]).any():
        raise ValueError("Afiliación duplicada por diputado y votación.")
    mapa = clasificacion_partidos(documento)
    a["grupo_bcall"] = a.diputado_id.map(grupos_bcall)
    # Independientes se mantiene sin grupo externo, incluso ante JSON mal codificado.
    externo = a.partido_nombre.map(mapa).astype("Int64")
    externo = externo.mask(a.partido_nombre.eq("Independientes"), pd.NA)
    a["codigo_externo"] = externo
    a["grupo_externo"] = externo.map({0: "L", 1: "R"})
    a["comparable"] = a.grupo_bcall.notna() & externo.notna()
    a["coincide"] = pd.Series(pd.NA, index=a.index, dtype="boolean")
    mask = a.comparable
    a.loc[mask, "coincide"] = a.loc[mask, "grupo_bcall"].eq(a.loc[mask, "grupo_externo"])
    a["razon_no_comparable"] = None
    a.loc[a.grupo_bcall.isna(), "razon_no_comparable"] = "SIN_RESULTADO_BCALL"
    a.loc[a.grupo_bcall.notna() & externo.isna(), "razon_no_comparable"] = (
        "SIN_ASIGNACION_EXTERNA"
    )
    return a
