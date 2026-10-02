"""B02: participación observada, celdas sin registro y conciliación de conteos.

Caracteriza lo que el corpus registra, sin reconstruir un padrón: no supone que todos
los diputados podían votar en todas las votaciones ni infiere ausencia o dispensa. Una
combinación diputado × votación sin decisión queda como "sin_registro", nunca como cero.
La afiliación es la que F3 ya integró en cada fila (partido vigente a la fecha); no se
repite el cruce de militancias. Los estados de calidad de militancias se toman tal cual
de reporte_calidad.csv de F3. La participación es la proporción de votaciones del corpus
con decisión sustantiva (la que usa el filtro de B-Call): no es tasa de asistencia.
B02 informa el filtro de B-Call, pero no elimina a nadie.
"""

import json
import tomllib
from pathlib import Path

import pandas as pd

from F4.src.analisis.corte import calcular_sha256

CLAVE = ["diputado_id", "votacion_id"]
CATEGORIAS = ["Sí", "No", "Abstención"]
TOTALES_F3 = {"Sí": "total_si", "No": "total_no", "Abstención": "total_abstencion"}
SIN_PARTIDO = "<sin_partido>"
RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]
REPORTES = "F4/data/reports"
SALIDAS = {
    "participacion": "participacion_observada.csv",
    "sin_registro": "celdas_sin_registro.csv",
    "partido_votacion": "conteos_partido_votacion.csv",
    "conciliacion": "conciliacion_conteos.json",
}


class ErrorCobertura(ValueError):
    """La tabla codificada no cumple lo que B02 necesita."""


class ParticipacionCorpus:
    """Conteos de B02 sobre votos_codificados (B01).

    votos: una fila por decisión registrada, con voto_nominal y observado de B01.
    calidad_f3: reporte_calidad.csv de F3; se usan sus filas de la tabla 'militancias'.
    umbral_bcall: [bcall].threshold de analisis.toml; se informa con comparación '>'.
    """

    def __init__(
        self,
        votos: pd.DataFrame,
        calidad_f3: pd.DataFrame | None = None,
        *,
        umbral_bcall: float = 0.10,
    ) -> None:
        requeridas = {*CLAVE, "fecha", "partido_id", "partido_alias", "voto_nominal",
                      "observado", "total_dispensado", *TOTALES_F3.values()}
        if faltan := sorted(requeridas - set(votos.columns)):
            raise ErrorCobertura(f"Faltan columnas de B01: {faltan}.")
        if votos[CLAVE].isna().any(axis=None) or votos.duplicated(CLAVE).any():
            raise ErrorCobertura("La clave diputado × votación debe ser completa y única.")
        observado = votos["observado"].astype("string").str.lower().eq("true")
        sustantivo = votos["voto_nominal"].isin(CATEGORIAS)
        if (observado != sustantivo).any():
            raise ErrorCobertura("'observado' no coincide con un voto nominal sustantivo.")
        self.votos = votos.loc[observado].copy()
        self.votaciones = (votos.groupby("votacion_id")["fecha"].first().reset_index()
                           .sort_values(["fecha", "votacion_id"]).reset_index(drop=True))
        self.calidad = calidad_f3
        self.umbral = float(umbral_bcall)

    # ------------------------------------------------------------------ afiliación F3
    def _calidad_militancia(self) -> pd.Series:
        """Tipos de observación de F3 sobre militancias, por diputado (sin recalcular)."""
        if self.calidad is None:
            return pd.Series(dtype="string", name="calidad_militancia_f3")
        mil = self.calidad[self.calidad["tabla"] == "militancias"]
        return (mil.groupby("diputado_id")["tipo"]
                .agg(lambda s: "|".join(sorted(s.dropna().unique())))
                .rename("calidad_militancia_f3"))

    def afiliacion(self) -> pd.DataFrame:
        """Afiliación de F3 por voto, su estado y las observaciones de calidad de F3."""
        base = self.votos[[*CLAVE, "partido_id", "partido_alias"]]
        salida = base.merge(self._calidad_militancia(), left_on="diputado_id",
                            right_index=True, how="left")
        salida["estado_afiliacion"] = (salida["partido_id"].isna()
                                       .map({True: "no_resuelta", False: "resuelta"}))
        return salida.reset_index(drop=True)

    # ----------------------------------------------------------------- participación
    def participacion_diputados(self) -> pd.DataFrame:
        """Decisiones sustantivas por diputado sobre el total de votaciones del corpus."""
        n_corpus = len(self.votaciones)
        conteo = (self.votos.groupby(["diputado_id", "voto_nominal"]).size()
                  .unstack(fill_value=0).reindex(columns=CATEGORIAS, fill_value=0))
        tabla = conteo.set_axis(["n_si", "n_no", "n_abst"], axis=1)
        tabla["n_votos_observados"] = conteo.sum(axis=1)
        tabla["n_votaciones_corpus"] = n_corpus
        tabla["n_sin_registro"] = n_corpus - tabla["n_votos_observados"]
        tabla["participacion_corpus"] = tabla["n_votos_observados"] / n_corpus
        tabla["supera_umbral_bcall"] = tabla["participacion_corpus"] > self.umbral
        partidos = (self.votos.groupby("diputado_id")["partido_alias"]
                    .agg(lambda s: "|".join(sorted(s.dropna().unique()))))
        tabla = tabla.join(partidos.rename("partidos_observados"))
        tabla = tabla.join(self._calidad_militancia())
        return (tabla.reset_index().sort_values("diputado_id", key=_orden_id)
                .reset_index(drop=True))

    def celdas_sin_registro(self) -> pd.DataFrame:
        """Diputado × votación sin decisión, entre los diputados presentes en el corpus.

        No dice por qué falta: ausencia, pareo, dispensa o no estar en ejercicio no son
        distinguibles con esta fuente.
        """
        diputados = pd.DataFrame({"diputado_id": self.votos["diputado_id"].unique()})
        grilla = diputados.merge(self.votaciones, how="cross")
        marcadas = grilla.merge(self.votos[CLAVE], on=CLAVE, how="left", indicator=True)
        faltan = marcadas[marcadas["_merge"] == "left_only"].drop(columns="_merge")
        return (faltan.assign(estado="sin_registro")
                .sort_values(CLAVE, key=_orden_id).reset_index(drop=True))

    # ------------------------------------------------------------- partido × votación
    def conteos_partido_votacion(self) -> pd.DataFrame:
        """Y, N, A y T observados por partido vigente × votación.

        Las filas sin partido se agrupan como <sin_partido> para que sigan visibles.
        """
        afil = self.afiliacion()
        votos = self.votos.merge(afil[[*CLAVE, "estado_afiliacion",
                                       "calidad_militancia_f3"]], on=CLAVE)
        votos[["partido_id", "partido_alias"]] = (
            votos[["partido_id", "partido_alias"]].fillna(SIN_PARTIDO))
        llaves = ["partido_id", "partido_alias", "votacion_id"]
        conteo = (votos.groupby([*llaves, "voto_nominal"]).size()
                  .unstack(fill_value=0).reindex(columns=CATEGORIAS, fill_value=0)
                  .set_axis(["Y", "N", "A"], axis=1))
        conteo["T"] = conteo[["Y", "N", "A"]].sum(axis=1)
        grupos = votos.groupby(llaves)
        conteo["n_afiliacion_no_resuelta"] = grupos["estado_afiliacion"].agg(
            lambda s: int((s == "no_resuelta").sum()))
        conteo["n_con_observacion_f3"] = grupos["calidad_militancia_f3"].agg(
            lambda s: int(s.notna().sum()))
        return conteo.reset_index().sort_values(llaves).reset_index(drop=True)

    # -------------------------------------------------------------------- conciliación
    def conciliacion(self) -> pd.DataFrame:
        """Conteos nominales por votación frente a los totales F3 de la misma votación."""
        obs = (self.votos.groupby(["votacion_id", "voto_nominal"]).size()
               .unstack(fill_value=0).reindex(columns=CATEGORIAS, fill_value=0))
        f3 = (self.votos.groupby("votacion_id")[[*TOTALES_F3.values(), "total_dispensado"]]
              .first().apply(pd.to_numeric))
        tabla = pd.DataFrame(index=obs.index)
        for cat, col in TOTALES_F3.items():
            tabla[f"obs_{col}"] = obs[cat]
            tabla[f"f3_{col}"] = f3[col]
            tabla[f"dif_{col}"] = obs[cat] - f3[col]
        tabla["f3_total_dispensado"] = f3["total_dispensado"]
        coincide = tabla.filter(like="dif_").eq(0).all(axis=1)
        tabla["estado"] = coincide.map({True: "coincide", False: "pendiente"})
        orden = self.votaciones.set_index("votacion_id").index
        return tabla.reindex(orden).reset_index()

    # ------------------------------------------------------------------------ reporte
    def reporte_conciliacion(self, trazabilidad: dict | None = None) -> dict:
        """Contenido de conciliacion_conteos.json: resumen, filtro y detalle."""
        part = self.participacion_diputados()
        conc = self.conciliacion()
        bajo = part[~part["supera_umbral_bcall"]]
        afil = self.afiliacion()
        return {
            "tarea": "B02",
            "trazabilidad": trazabilidad or {},
            "resumen": {
                "n_votaciones_corpus": len(self.votaciones),
                "n_diputados_en_corpus": int(len(part)),
                "n_decisiones_observadas": int(len(self.votos)),
                "n_celdas_sin_registro": int(len(self.celdas_sin_registro())),
                "n_afiliacion_no_resuelta": int((afil["estado_afiliacion"]
                                                 == "no_resuelta").sum()),
                "n_votos_con_observacion_f3": int(afil["calidad_militancia_f3"]
                                                  .notna().sum()),
                "distribucion_votos_por_diputado": {
                    str(k): int(v) for k, v in
                    part["n_votos_observados"].value_counts().sort_index().items()},
            },
            "filtro_bcall_informado": {
                "umbral": self.umbral, "comparacion": ">",
                "denominador": "votaciones del corpus",
                "n_no_superan": int(len(bajo)),
                "diputados_no_superan": bajo["diputado_id"].astype(str).tolist(),
                "nota": "B02 no elimina personas; el filtro lo aplica B-Call después "
                        "del agrupamiento automático.",
            },
            "conciliacion": {
                "votaciones_coinciden": int((conc["estado"] == "coincide").sum()),
                "votaciones_pendientes": int((conc["estado"] == "pendiente").sum()),
                "detalle": conc.to_dict("records"),
            },
            "advertencia": "Participación = decisiones sustantivas / votaciones del corpus. "
                           "No es tasa de asistencia: no hay padrón verificable de habilitados.",
        }

    def exportar(self, directorio: Path | str, trazabilidad: dict | None = None) -> dict:
        """Escribe las cuatro salidas de B02 de forma determinista."""
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)
        tablas = {
            "participacion": self.participacion_diputados(),
            "sin_registro": self.celdas_sin_registro(),
            "partido_votacion": self.conteos_partido_votacion(),
        }
        rutas = {}
        for clave, tabla in tablas.items():
            rutas[clave] = directorio / SALIDAS[clave]
            tabla.to_csv(rutas[clave], index=False, lineterminator="\n")
        rutas["conciliacion"] = directorio / SALIDAS["conciliacion"]
        texto = json.dumps(self.reporte_conciliacion(trazabilidad), ensure_ascii=False,
                           indent=2, default=str)
        rutas["conciliacion"].write_text(texto + "\n", encoding="utf-8")
        return rutas


def _orden_id(serie: pd.Series) -> pd.Series:
    """Orden numérico de IDs guardados como texto, estable entre plataformas."""
    return pd.to_numeric(serie, errors="coerce").fillna(float("inf"))


def desde_repositorio(raiz: Path | str = RAIZ_REPOSITORIO) -> tuple[ParticipacionCorpus, dict]:
    """Carga B01, la calidad de F3 y el umbral de A03; verifica el corte de A01."""
    raiz = Path(raiz)
    with (raiz / "F4/config/analisis.toml").open("rb") as archivo:
        umbral = tomllib.load(archivo)["bcall"]["threshold"]
    ruta_votos = raiz / "F4/data/processed/votos_codificados.csv"
    ruta_calidad = raiz / "F3/data/processed/reporte_calidad.csv"
    manifiesto = json.loads((raiz / f"{REPORTES}/manifiesto_corte.json")
                            .read_text(encoding="utf-8"))
    votos = pd.read_csv(ruta_votos, dtype="string")
    corte = manifiesto["entrada"]
    mismo_corte = (len(votos) == corte["filas"]
                   and votos["votacion_id"].nunique() == corte["votaciones"])
    if not mismo_corte:
        raise ErrorCobertura("votos_codificados no tiene las filas y votaciones del corte A01.")
    trazabilidad = {
        "corte_sha256": corte["sha256"],
        "corte_commit": manifiesto["reproducibilidad"]["commit_git"],
        "votos_codificados_sha256": calcular_sha256(ruta_votos),
        "reporte_calidad_f3_sha256": calcular_sha256(ruta_calidad),
        "filas_y_votaciones_coinciden_con_corte": mismo_corte,
    }
    corpus = ParticipacionCorpus(votos, pd.read_csv(ruta_calidad, dtype="string"),
                                 umbral_bcall=umbral)
    return corpus, trazabilidad


def main(raiz: Path | str = RAIZ_REPOSITORIO) -> dict:
    corpus, trazabilidad = desde_repositorio(raiz)
    rutas = corpus.exportar(Path(raiz) / REPORTES, trazabilidad)
    r = corpus.reporte_conciliacion(trazabilidad)
    print(f"B02: {r['resumen']['n_decisiones_observadas']} decisiones, "
          f"{r['resumen']['n_celdas_sin_registro']} celdas sin registro, "
          f"{r['filtro_bcall_informado']['n_no_superan']} bajo el filtro de B-Call, "
          f"{r['conciliacion']['votaciones_pendientes']} votaciones pendientes de conciliar.")
    return rutas


if __name__ == "__main__":
    main()
