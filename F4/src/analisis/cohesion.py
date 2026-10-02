"""Índices de cohesión por partido × votación.

Agreement Index (Hix, Noury y Roland, 2005) como índice principal; índice de Rice (1925)
y cohesión entrópica normalizada (Shannon, 1948) como contrastes. Se calculan sobre
matriz_nominal.csv y afiliacion_por_votacion.csv. Responden cuán unido vota un grupo,
no hacia dónde vota. Cada fila muestra Y, N, A y T; un valor no estimable es NA con
razón, nunca cero. Los parámetros de publicación vienen de [cohesion] en analisis.toml.

Criterios:
- Independientes (IND) no son un partido: se calculan, pero se marcan con
  tipo_grupo = "independientes" y no se publican como cohesión partidaria.
- Las votaciones unánimes en la sala se conservan, porque siguen informando sobre la
  unidad de cada grupo, y se marcan para poder comparar resultados con y sin ellas.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIAS = {"Sí": "Y", "No": "N", "Abstención": "A"}
LLAVES = ["partido_id", "partido_alias", "votacion_id"]
RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]
SALIDA = "F4/data/results/partidos/cohesion_por_votacion.csv"
SALIDA_RESUMEN = "F4/data/results/partidos/resumen_cohesion.csv"
INDICES = {
    "ai": ("agreement_index", "publicable_ai_entropia"),
    "rice": ("rice", "publicable_rice"),
    "entropia": ("cohesion_entropica", "publicable_ai_entropia"),
}


class ErrorCohesion(ValueError):
    """La entrada no permite calcular índices sin violar el contrato."""


# ------------------------------------------------------------------------ fórmulas
def agreement_index(y, n, a) -> np.ndarray:
    """AI = [máx(Y,N,A) − ½(T − máx)] / T; NA si T = 0."""
    y, n, a = (np.asarray(v, dtype=float) for v in (y, n, a))
    t = y + n + a
    mx = np.maximum.reduce([y, n, a])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, (mx - 0.5 * (t - mx)) / t, np.nan)


def rice(y, n) -> np.ndarray:
    """Rice = |Y − N| / (Y + N); la abstención no entra. NA si Y + N = 0."""
    y, n = np.asarray(y, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y + n > 0, np.abs(y - n) / (y + n), np.nan)


def cohesion_entropica(y, n, a, k: int = 3) -> np.ndarray:
    """C^H = 1 − H/ln K con H de Shannon (ln natural, 0·ln0 = 0, K fijo). NA si T = 0."""
    conteos = np.column_stack([np.asarray(v, dtype=float) for v in (y, n, a)])
    t = conteos.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(t > 0, conteos / t, 0.0)
        termino = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0)
    h = -termino.sum(axis=1)
    return np.where(t[:, 0] > 0, 1 - h / np.log(k), np.nan)


@dataclass(frozen=True)
class ParametrosCohesion:
    """Reglas de [cohesion] en analisis.toml más el criterio sobre independientes."""

    min_decisiones_publicacion: int = 2
    min_binarios_publicacion_rice: int = 2
    categorias_entropia: int = 3
    min_votaciones_resumen: int = 2
    grupos_no_partidarios: tuple[str, ...] = ("IND",)

    @classmethod
    def desde_toml(cls, ruta: Path | str) -> "ParametrosCohesion":
        with Path(ruta).open("rb") as archivo:
            c = tomllib.load(archivo)["cohesion"]
        return cls(
            min_decisiones_publicacion=int(c["min_decisiones_publicacion_partido_votacion"]),
            min_binarios_publicacion_rice=int(c["min_votos_binarios_publicacion_rice"]),
            categorias_entropia=int(c["categorias_entropia"]),
            min_votaciones_resumen=int(c["min_votaciones_resumen_principal"]),
        )


class IndicesCohesion:
    """Calcula AI, Rice y C^H por partido vigente × votación."""

    def __init__(self, parametros: ParametrosCohesion | None = None) -> None:
        self.p = parametros or ParametrosCohesion()

    @staticmethod
    def votos_largos(nominal: pd.DataFrame, afiliacion: pd.DataFrame) -> pd.DataFrame:
        """Une la matriz nominal con la afiliación por votación (solo celdas con voto)."""
        largo = nominal.melt(id_vars="diputado_id", var_name="votacion_id",
                             value_name="voto_nominal").dropna(subset=["voto_nominal"])
        clave = ["diputado_id", "votacion_id"]
        afil = afiliacion.astype({c: "string" for c in clave})
        largo = largo.astype({c: "string" for c in clave})
        unido = largo.merge(afil, on=clave, how="left", validate="one_to_one",
                            indicator=True)
        if (unido["_merge"] != "both").any():
            raise ErrorCohesion("Hay votos observados sin fila en afiliacion_por_votacion.")
        return unido.drop(columns="_merge")

    def conteos(self, votos: pd.DataFrame) -> pd.DataFrame:
        """Y, N, A, T por partido × votación desde una tabla larga de votos sustantivos."""
        if not votos["voto_nominal"].isin(list(CATEGORIAS)).all():
            raise ErrorCohesion("voto_nominal solo admite Sí, No y Abstención.")
        votos = votos.copy()
        votos[["partido_id", "partido_alias"]] = (
            votos[["partido_id", "partido_alias"]].fillna("<sin_partido>"))
        tabla = (votos.groupby([*LLAVES, "voto_nominal"]).size().unstack(fill_value=0)
                 .reindex(columns=list(CATEGORIAS), fill_value=0)
                 .rename(columns=CATEGORIAS).rename_axis(columns=None))
        tabla["T"] = tabla[["Y", "N", "A"]].sum(axis=1)
        fechas = votos.groupby("votacion_id")["fecha"].first()
        return tabla.reset_index().assign(fecha=lambda d: d["votacion_id"].map(fechas))

    def calcular(self, conteos: pd.DataFrame) -> pd.DataFrame:
        """Índices, razones de NA y marcas de publicación por partido × votación."""
        t = conteos.copy()
        y, n, a = t["Y"], t["N"], t["A"]
        t["votos_binarios"] = y + n
        t["agreement_index"] = agreement_index(y, n, a)
        t["rice"] = rice(y, n)
        t["cohesion_entropica"] = cohesion_entropica(y, n, a, self.p.categorias_entropia)
        t["razon_na_ai_entropia"] = _motivo(t["T"] == 0, "T_cero")
        t["razon_na_rice"] = _motivo(t["votos_binarios"] == 0, "sin_si_ni_no")

        no_partido = t["partido_id"].isin([*self.p.grupos_no_partidarios, "<sin_partido>"])
        t["tipo_grupo"] = np.where(t["partido_id"].isin(self.p.grupos_no_partidarios),
                                   "independientes",
                                   np.where(t["partido_id"] == "<sin_partido>",
                                            "sin_partido", "partido"))
        pocos = t["T"] < self.p.min_decisiones_publicacion
        pocos_bin = t["votos_binarios"] < self.p.min_binarios_publicacion_rice
        t["publicable_ai_entropia"] = ~no_partido & ~pocos
        t["publicable_rice"] = ~no_partido & ~pocos_bin
        t["razon_no_publicable"] = _razones(no_partido, pocos, pocos_bin, t["tipo_grupo"])

        sala = t.groupby("votacion_id")[["Y", "N", "A"]].sum()
        unanimes = sala.index[(sala > 0).sum(axis=1) == 1]
        t["votacion_unanime_sala"] = t["votacion_id"].isin(unanimes)

        columnas = [*LLAVES, "fecha", "tipo_grupo", "Y", "N", "A", "T", "votos_binarios",
                    "agreement_index", "rice", "cohesion_entropica", "razon_na_ai_entropia",
                    "razon_na_rice", "publicable_ai_entropia", "publicable_rice",
                    "razon_no_publicable", "votacion_unanime_sala"]
        return (t[columnas].sort_values(["partido_id", "fecha", "votacion_id"])
                .reset_index(drop=True))


class ResumenCohesion:
    """Resume la cohesión por partido sin esconder la variación entre votaciones.

    Usa la mediana no ponderada de las votaciones publicables de cada índice, junto con
    el rango intercuartílico, el mínimo, el máximo y el número de votaciones válidas.
    No ordena por cohesión ni asigna etiquetas de "alta" o "baja".
    """

    def __init__(self, parametros: ParametrosCohesion | None = None) -> None:
        self.p = parametros or ParametrosCohesion()

    def calcular(self, por_votacion: pd.DataFrame) -> pd.DataFrame:
        """Una fila por partido o grupo, a partir de la tabla de cohesión por votación."""
        grupos = por_votacion.groupby(["partido_id", "partido_alias", "tipo_grupo"],
                                      sort=True)
        filas = [self._resumir(llave, g) for llave, g in grupos]
        return pd.DataFrame(filas).reset_index(drop=True)

    def _resumir(self, llave: tuple, g: pd.DataFrame) -> dict:
        partido_id, partido_alias, tipo_grupo = llave
        fila = {"partido_id": partido_id, "partido_alias": partido_alias,
                "tipo_grupo": tipo_grupo, "n_votaciones_observadas": len(g)}
        for nombre, (columna, publicable) in INDICES.items():
            valores = g.loc[g[publicable].astype(bool), columna].astype(float)
            fila[f"n_validas_{nombre}"] = len(valores)
            fila[f"mediana_{nombre}"] = valores.median() if len(valores) else np.nan
            fila[f"riq_{nombre}"] = (valores.quantile(0.75) - valores.quantile(0.25)
                                     if len(valores) else np.nan)
            fila[f"min_{nombre}"] = valores.min() if len(valores) else np.nan
            fila[f"max_{nombre}"] = valores.max() if len(valores) else np.nan
        sin_unanimes = g.loc[g["publicable_ai_entropia"].astype(bool)
                             & ~g["votacion_unanime_sala"].astype(bool), "agreement_index"]
        fila["n_validas_ai_sin_unanimes"] = len(sin_unanimes)
        fila["mediana_ai_sin_unanimes"] = (sin_unanimes.astype(float).median()
                                           if len(sin_unanimes) else np.nan)
        fila["bancada_mediana_T"] = float(g["T"].median())
        fila["bancada_max_T"] = int(g["T"].max())
        for c in ("Y", "N", "A", "T"):
            fila[f"total_{c}"] = int(g[c].sum())
        excluidas = g.loc[~g["publicable_ai_entropia"].astype(bool), "razon_no_publicable"]
        fila["n_votaciones_excluidas_ai"] = len(excluidas)
        fila["razones_exclusion"] = ("|".join(sorted(excluidas.dropna().unique()))
                                     or pd.NA)
        motivo = []
        if tipo_grupo != "partido":
            motivo.append(f"grupo_{tipo_grupo}")
        elif fila["n_validas_ai"] < self.p.min_votaciones_resumen:
            motivo.append("votaciones_validas_bajo_minimo")
        fila["publicable_resumen"] = not motivo
        fila["razon_no_publicable_resumen"] = "|".join(motivo) or pd.NA
        return fila


def _motivo(condicion: pd.Series, texto: str) -> pd.Series:
    """Motivo de NA donde se cumple la condición; NA (no texto vacío) en el resto."""
    return pd.Series(pd.NA, index=condicion.index, dtype="string").mask(condicion, texto)


def _razones(no_partido, pocos, pocos_bin, tipo) -> pd.Series:
    """Texto con todas las razones por las que una fila no se publica (o vacío)."""
    partes = [
        np.where(no_partido, "grupo_" + tipo.astype(str), ""),
        np.where(pocos, "T_bajo_minimo", ""),
        np.where(pocos_bin & ~pocos, "binarios_bajo_minimo_rice", ""),
    ]
    texto = pd.Series(["|".join(p for p in fila if p) for fila in zip(*partes)],
                      index=tipo.index, dtype="string")
    return texto.replace("", pd.NA)


def desde_repositorio(raiz: Path | str = RAIZ_REPOSITORIO) -> pd.DataFrame:
    """Lee la matriz nominal, la afiliación y analisis.toml; devuelve la tabla de cohesión."""
    raiz = Path(raiz)
    modelo = IndicesCohesion(ParametrosCohesion.desde_toml(raiz / "F4/config/analisis.toml"))
    nominal = pd.read_csv(raiz / "F4/data/processed/matriz_nominal.csv", dtype="string")
    afil = pd.read_csv(raiz / "F4/data/processed/afiliacion_por_votacion.csv", dtype="string")
    return modelo.calcular(modelo.conteos(IndicesCohesion.votos_largos(nominal, afil)))


def main(raiz: Path | str = RAIZ_REPOSITORIO) -> tuple[Path, Path]:
    """Escribe la cohesión por votación y su resumen por partido."""
    raiz = Path(raiz)
    tabla = desde_repositorio(raiz)
    parametros = ParametrosCohesion.desde_toml(raiz / "F4/config/analisis.toml")
    resumen = ResumenCohesion(parametros).calcular(tabla)
    destino, destino_resumen = raiz / SALIDA, raiz / SALIDA_RESUMEN
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False, lineterminator="\n")
    resumen.to_csv(destino_resumen, index=False, lineterminator="\n")
    print(f"Cohesión: {len(tabla)} filas partido × votación; "
          f"{int(tabla['publicable_ai_entropia'].sum())} publicables (AI/entropía), "
          f"{int(tabla['publicable_rice'].sum())} publicables (Rice).")
    print(f"Resumen: {len(resumen)} grupos; "
          f"{int(resumen['publicable_resumen'].sum())} partidos con resumen publicable.")
    return destino, destino_resumen


if __name__ == "__main__":
    main()
