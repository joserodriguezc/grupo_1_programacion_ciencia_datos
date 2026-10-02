"""Auditoría de claves y dominios del corte de F3 antes de estimar.

Solo lee. No corrige, no recodifica ni filtra: cada regla termina en ok, advertencia
explicada o error. Una advertencia exige explicación; lo que no se puede explicar es
error y bloquea la entrada. El tratamiento de cada caso se decide al codificar los
votos y al caracterizar la participación, no aquí.

Verifica el corte contra manifiesto_corte.json sin regenerarlo. Lee el CSV como texto,
no con corte.cargar_entrada(), para auditar los códigos tal como vienen de F3 (una
inferencia numérica ocultaría, por ejemplo, un código "01" leído como 1). Incluye la
validación de unicidad de la clave diputado × votación (regla R03).
"""

import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from F4.src.analisis.corte import calcular_sha256

Estado = Literal["ok", "advertencia", "error"]

CLAVE = ["diputado_id", "votacion_id"]
CODIGOS_VOTO = {"1": "Afirmativo", "0": "En Contra", "2": "Abstención"}
TOTALES = {"1": "total_si", "0": "total_no", "2": "total_abstencion"}
ATRIBUTOS_VOTACION = [
    "fecha", "total_si", "total_no", "total_abstencion", "total_dispensado",
    "quorum_codigo", "resultado_codigo", "tipo_votacion_proyecto_ley", "numero_boletin",
]
ATRIBUTOS_DIPUTADO = ["nombre", "apellido_paterno", "apellido_materno",
                      "fecha_nacimiento", "sexo_valor"]
OBLIGATORIAS = CLAVE + ["fecha", "opcion_codigo", "opcion_voto", "total_si", "total_no",
                        "total_abstencion", "total_dispensado", "numero_boletin",
                        "periodo_id", "partido_id", "partido_nombre", "partido_alias"]
MAX_EJEMPLOS = 10


@dataclass(frozen=True, slots=True)
class Hallazgo:
    """Resultado de una regla; una advertencia sin explicación no es admisible."""

    regla: str
    estado: Estado
    detalle: str
    explicacion: str | None = None
    n_afectados: int = 0
    ejemplos: tuple = ()

    def __post_init__(self) -> None:
        if self.estado not in ("ok", "advertencia", "error"):
            raise ValueError(f"Estado desconocido: {self.estado!r}")
        if self.estado == "advertencia" and not self.explicacion:
            raise ValueError(f"La advertencia {self.regla} requiere explicación.")


@dataclass(frozen=True)
class ReporteAuditoria:
    entrada: dict
    hallazgos: tuple[Hallazgo, ...]
    perfil: dict = field(default_factory=dict)

    @property
    def aprobado(self) -> bool:
        return not any(h.estado == "error" for h in self.hallazgos)

    def a_dict(self) -> dict:
        conteo = {e: sum(h.estado == e for h in self.hallazgos)
                  for e in ("ok", "advertencia", "error")}
        return {
            "reporte": "auditoria_entrada",
            "estado_entrada": "aprobada" if self.aprobado else "bloqueada",
            "resumen": conteo,
            **self.entrada,
            "hallazgos": [asdict(h) for h in self.hallazgos],
            "perfil": self.perfil,
        }


def _ejemplos(df: pd.DataFrame | pd.Index) -> tuple:
    filas = df.to_frame(index=False) if isinstance(df, pd.Index) else df
    return tuple(filas.head(MAX_EJEMPLOS).astype(str).to_dict("records"))


class AuditorEntrada:
    """Aplica las reglas de auditoría a la tabla analítica y la contrasta con F3.

    Todas las tablas se reciben como texto (dtype string) para no alterar códigos.
    Las tablas de contraste son opcionales; si faltan, la regla se informa como error.
    """

    def __init__(
        self,
        tabla: pd.DataFrame,
        *,
        esperado: Mapping[str, int],
        detalle: pd.DataFrame | None = None,
        proyecto: pd.DataFrame | None = None,
        militancias: pd.DataFrame | None = None,
        padron: pd.DataFrame | None = None,
        calidad_f3: pd.DataFrame | None = None,
        manifiesto: Mapping | None = None,
        sha256_tabla: str | None = None,
    ) -> None:
        self.t = tabla
        self.esperado = dict(esperado)
        self.detalle, self.proyecto = detalle, proyecto
        self.militancias, self.padron, self.calidad = militancias, padron, calidad_f3
        self.manifiesto, self.sha = manifiesto, sha256_tabla

    # ----------------------------------------------------------------- orquestación
    def auditar(self, entrada: Mapping | None = None) -> ReporteAuditoria:
        faltan = [c for c in OBLIGATORIAS if c not in self.t.columns]
        hallazgos = [self._columnas(faltan)]
        if not faltan:  # sin columnas obligatorias las demás reglas no son evaluables
            for regla in (
                self._manifiesto, self._dimensiones, self._clave, self._tipos,
                self._dominio_voto, self._atributos_votacion, self._totales,
                self._boletin, self._detalle, self._proyecto, self._militancia,
                self._atributos_diputado, self._padron, self._nulos, self._anomalias_f3,
            ):
                hallazgos.extend(regla())
        perfil = self.perfil() if not faltan else {}
        return ReporteAuditoria(dict(entrada or {}), tuple(hallazgos), perfil)

    # ----------------------------------------------------------------------- reglas
    def _columnas(self, faltan: list[str]) -> Hallazgo:
        if faltan:
            return Hallazgo("R02_columnas", "error", f"Faltan columnas: {faltan}.",
                            n_afectados=len(faltan))
        return Hallazgo("R02_columnas", "ok", "Columnas obligatorias presentes.")

    def _manifiesto(self) -> list[Hallazgo]:
        """Compara el archivo leído con manifiesto_corte.json; nunca lo regenera."""
        if not self.manifiesto:
            return [Hallazgo("R00_manifiesto_corte", "error",
                             "Sin manifiesto_corte.json: commit y hash "
                             "del corte no están congelados.")]
        m = self.manifiesto.get("entrada", {})
        observado = {"sha256": self.sha, "filas": len(self.t), "columnas": self.t.shape[1],
                     "votaciones": self.t["votacion_id"].nunique()}
        dif = {k: {"leido": v, "manifiesto": m.get(k)}
               for k, v in observado.items() if m.get(k) != v}
        estado = self.manifiesto.get("validacion", {}).get("estado")
        if estado != "ok":
            dif["validacion.estado"] = {"leido": "ok", "manifiesto": estado}
        if dif:
            return [Hallazgo("R00_manifiesto_corte", "error",
                             f"El archivo leído no corresponde al corte congelado: {dif}.",
                             n_afectados=len(dif))]
        commit = self.manifiesto.get("reproducibilidad", {}).get("commit_git")
        return [Hallazgo("R00_manifiesto_corte", "ok",
                         f"SHA256, filas, columnas y votaciones coinciden con el manifiesto "
                         f"(commit {commit}).")]

    def _dimensiones(self) -> list[Hallazgo]:
        obs = {"filas": len(self.t), "columnas": self.t.shape[1],
               "votaciones": self.t["votacion_id"].nunique()}
        dif = {k: (obs[k], v) for k, v in self.esperado.items() if obs.get(k) != v}
        if dif:
            return [Hallazgo("R01_dimensiones", "error",
                             f"Observado vs esperado distinto: {dif}.", n_afectados=len(dif))]
        return [Hallazgo("R01_dimensiones", "ok", f"Dimensiones esperadas: {obs}.")]

    def _clave(self) -> list[Hallazgo]:
        nulos = self.t[CLAVE].isna().any(axis=1)
        dup = self.t.duplicated(CLAVE, keep=False) & ~nulos
        if nulos.any() or dup.any():
            return [Hallazgo("R03_clave_unica", "error",
                             f"{int(nulos.sum())} filas con clave nula y {int(dup.sum())} "
                             "filas duplicadas diputado × votación.",
                             n_afectados=int(nulos.sum() + dup.sum()),
                             ejemplos=_ejemplos(self.t.loc[dup | nulos, CLAVE]))]
        return [Hallazgo("R03_clave_unica", "ok", "Clave diputado × votación única y completa.")]

    def _tipos(self) -> list[Hallazgo]:
        malos = {}
        for col in CLAVE + ["total_si", "total_no", "total_abstencion", "total_dispensado"]:
            num = pd.to_numeric(self.t[col], errors="coerce")
            invalido = num.isna() | (num < 0) | (num % 1 != 0)
            if invalido.any():
                malos[col] = int(invalido.sum())
        fechas = pd.to_datetime(self.t["fecha"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
        if fechas.isna().any():
            malos["fecha"] = int(fechas.isna().sum())
        if malos:
            return [Hallazgo("R04_tipos", "error", f"Valores no válidos por columna: {malos}.",
                             n_afectados=sum(malos.values()))]
        return [Hallazgo("R04_tipos", "ok",
                         "IDs y totales enteros no negativos; fechas con formato válido.")]

    def _dominio_voto(self) -> list[Hallazgo]:
        pares = self.t[["opcion_codigo", "opcion_voto"]].drop_duplicates()
        fuera = pares[pares.apply(
            lambda f: CODIGOS_VOTO.get(f.opcion_codigo) != f.opcion_voto, axis=1)]
        if len(fuera) or self.t["opcion_codigo"].isna().any():
            return [Hallazgo("R05_dominio_voto", "error",
                             "Códigos de voto fuera del diccionario o sin correspondencia "
                             "uno a uno con su texto.", n_afectados=len(fuera),
                             ejemplos=_ejemplos(fuera))]
        return [Hallazgo("R05_dominio_voto", "ok",
                         f"Solo códigos {sorted(pares.opcion_codigo)} con texto consistente; "
                         "no hay registros nominales de ausencia ni dispensa.")]

    def _atributos_votacion(self) -> list[Hallazgo]:
        var = self.t.groupby("votacion_id")[ATRIBUTOS_VOTACION].nunique(dropna=False)
        malas = var[(var > 1).any(axis=1)]
        if len(malas):
            return [Hallazgo("R06_atributos_votacion", "error",
                             "Atributos de votación que cambian dentro de una misma votación.",
                             n_afectados=len(malas), ejemplos=_ejemplos(malas.reset_index()))]
        return [Hallazgo("R06_atributos_votacion", "ok",
                         "Fecha, totales, quórum y tipo son constantes por votación.")]

    def _totales(self) -> list[Hallazgo]:
        obs = (self.t.groupby(["votacion_id", "opcion_codigo"]).size()
               .unstack(fill_value=0).reindex(columns=list(TOTALES), fill_value=0))
        dec = (self.t.groupby("votacion_id")[list(TOTALES.values())].first()
               .apply(pd.to_numeric).set_axis(list(TOTALES), axis=1))
        dif = obs.ne(dec).any(axis=1)
        salida = []
        if dif.any():
            tabla = obs[dif].add_prefix("obs_").join(dec[dif].add_prefix("decl_"))
            salida.append(Hallazgo("R07_totales", "error",
                                   "Conteo nominal distinto de los totales declarados.",
                                   n_afectados=int(dif.sum()),
                                   ejemplos=_ejemplos(tabla.reset_index())))
        else:
            salida.append(Hallazgo("R07_totales", "ok",
                                   "Sí/No/Abstención nominales = totales en las "
                                   f"{len(obs)} votaciones."))
        disp = pd.to_numeric(self.t["total_dispensado"]).gt(0)
        if disp.any():
            salida.append(Hallazgo(
                "R07_dispensas", "advertencia",
                "Votaciones con dispensas declaradas sin registro nominal.",
                "La fuente informa el total pero no quién fue dispensado.",
                n_afectados=self.t.loc[disp, "votacion_id"].nunique()))
        return salida

    def _boletin(self) -> list[Hallazgo]:
        boletines = sorted(self.t["numero_boletin"].dropna().unique())
        if len(boletines) != 1:
            return [Hallazgo("R08_boletin", "error",
                             f"Se esperaba un solo proyecto; hay {boletines}.")]
        return [Hallazgo("R08_boletin", "ok", f"Un solo boletín: {boletines[0]}.")]

    def _detalle(self) -> list[Hallazgo]:
        if self.detalle is None:
            return [Hallazgo("R09_contraste_detalle", "error",
                             "No se entregó detalle_votaciones_procesado para contrastar.")]
        cols = CLAVE + ["opcion_codigo"]
        m = self.t[cols].merge(self.detalle[cols], on=CLAVE, how="outer",
                               suffixes=("", "_f3"), indicator=True)
        malas = m[(m["_merge"] != "both") | (m["opcion_codigo"] != m["opcion_codigo_f3"])]
        if len(malas):
            return [Hallazgo("R09_contraste_detalle", "error",
                             "Claves o votos distintos entre la tabla analítica y el detalle F3.",
                             n_afectados=len(malas), ejemplos=_ejemplos(malas))]
        return [Hallazgo("R09_contraste_detalle", "ok",
                         "Mismas claves y mismo voto que detalle_votaciones_procesado.")]

    def _proyecto(self) -> list[Hallazgo]:
        if self.proyecto is None:
            return [Hallazgo("R10_contraste_proyecto", "error",
                             "No se entregó proyecto_ley_procesado para contrastar.")]
        f3 = self.proyecto.rename(columns={"Id": "votacion_id", "TotalSi": "total_si",
                                           "TotalNo": "total_no",
                                           "TotalAbstencion": "total_abstencion"})
        cols = ["votacion_id", "total_si", "total_no", "total_abstencion"]
        propia = self.t[cols].drop_duplicates()
        m = propia.merge(f3[cols], on="votacion_id", how="outer",
                         suffixes=("", "_f3"), indicator=True)
        dist = (m["_merge"] != "both")
        for c in cols[1:]:
            dist |= m[c] != m[f"{c}_f3"]
        if dist.any():
            return [Hallazgo("R10_contraste_proyecto", "error",
                             "Votaciones o totales distintos de proyecto_ley_procesado.",
                             n_afectados=int(dist.sum()), ejemplos=_ejemplos(m[dist]))]
        return [Hallazgo("R10_contraste_proyecto", "ok",
                         f"Las {len(propia)} votaciones y sus totales coinciden con F3.")]

    def _militancia(self) -> list[Hallazgo]:
        salida = []
        sin = self.t["partido_id"].isna()
        if sin.any():
            salida.append(Hallazgo(
                "R11_militancia_nula", "advertencia",
                "Filas sin partido asignado.",
                "Se conservan para el análisis individual y se excluyen con razón "
                "de los agregados partidarios.", n_afectados=int(sin.sum())))
        if self.militancias is None:
            salida.append(Hallazgo("R11_militancia_fecha", "error",
                                   "No se entregó militancias_analiticas para contrastar."))
            return salida
        mil = self.militancias.assign(
            fi=pd.to_datetime(self.militancias["fecha_inicio"]),
            ft=pd.to_datetime(self.militancias["fecha_termino"]))
        votos = self.t[CLAVE + ["fecha", "partido_id"]].assign(f=pd.to_datetime(self.t["fecha"]))
        cruce = votos.merge(mil[["diputado_id", "partido_id", "fi", "ft"]],
                            on="diputado_id", suffixes=("", "_vigente"))
        cruce = cruce[(cruce.f >= cruce.fi) & (cruce.f <= cruce.ft)]
        vigente = cruce.groupby(CLAVE)["partido_id_vigente"].agg(
            lambda s: "|".join(sorted(set(s))))
        j = votos.set_index(CLAVE).join(vigente)
        malas = j[j["partido_id"].fillna("") != j["partido_id_vigente"].fillna("")]
        if len(malas):
            salida.append(Hallazgo(
                "R11_militancia_fecha", "error",
                "Partido de la fila distinto del vigente a la fecha, ausente o ambiguo "
                "(más de un intervalo vigente).", n_afectados=len(malas),
                ejemplos=_ejemplos(malas.reset_index()[CLAVE + ["partido_id",
                                                                 "partido_id_vigente"]])))
        else:
            salida.append(Hallazgo("R11_militancia_fecha", "ok",
                                   "Cada fila tiene un único partido vigente a la fecha "
                                   "y coincide con militancias_analiticas."))
        cambian = self.t.groupby("diputado_id")["partido_id"].nunique()
        cambian = cambian[cambian > 1]
        if len(cambian):
            salida.append(Hallazgo(
                "R11_cambio_militancia", "advertencia",
                "Diputados con distinto partido entre votaciones del corte.",
                "Cambios registrados en F3 y verificados en R11_militancia_fecha; los "
                "agregados partidarios usan el partido vigente en cada votación.",
                n_afectados=len(cambian), ejemplos=_ejemplos(cambian.index)))
        return salida

    def _atributos_diputado(self) -> list[Hallazgo]:
        var = self.t.groupby("diputado_id")[ATRIBUTOS_DIPUTADO].nunique(dropna=False)
        malas = var[(var > 1).any(axis=1)]
        if len(malas):
            return [Hallazgo("R12_atributos_diputado", "error",
                             "Un mismo diputado_id con distintos datos personales.",
                             n_afectados=len(malas), ejemplos=_ejemplos(malas.reset_index()))]
        return [Hallazgo("R12_atributos_diputado", "ok",
                         "Nombre, fecha de nacimiento y sexo constantes por diputado.")]

    def _padron(self) -> list[Hallazgo]:
        if self.padron is None:
            return [Hallazgo("R13_padron", "error",
                             "No se entregó diputados_procesados para contrastar.")]
        corte, padron = set(self.t["diputado_id"]), set(self.padron["diputado_id"])
        salida = []
        if fuera := sorted(corte - padron):
            salida.append(Hallazgo("R13_padron", "error",
                                   "Diputados con votos que no están en el padrón F3.",
                                   n_afectados=len(fuera), ejemplos=tuple(fuera[:MAX_EJEMPLOS])))
        else:
            salida.append(Hallazgo("R13_padron", "ok",
                                   "Todo diputado con votos está en el padrón F3."))
        if sin_votos := sorted(padron - corte):
            salida.append(Hallazgo(
                "R13_padron_sin_votos", "advertencia",
                "Diputados del padrón F3 sin ninguna fila en el corte.",
                "La tabla solo registra votos emitidos; su elegibilidad por votación "
                "(ausencia, reemplazo o fuera de ejercicio) no se infiere en esta auditoría.",
                n_afectados=len(sin_votos), ejemplos=tuple(sin_votos[:MAX_EJEMPLOS])))
        return salida

    def _nulos(self) -> list[Hallazgo]:
        salida = []
        nulos = self.t.isna().sum()
        nulos = nulos[(nulos > 0) & ~nulos.index.isin(OBLIGATORIAS)]
        for col, n in nulos.items():
            explicacion = self._explicar_nulo(col)
            if explicacion:
                salida.append(Hallazgo(f"R14_nulos_{col}", "advertencia",
                                       f"{n} valores nulos en {col}.", explicacion,
                                       n_afectados=int(n)))
            else:
                salida.append(Hallazgo(f"R14_nulos_{col}", "error",
                                       f"{n} valores nulos en {col} sin patrón explicable.",
                                       n_afectados=int(n)))
        obligatorias = self.t[[c for c in OBLIGATORIAS if c not in CLAVE]].isna().sum()
        for col, n in obligatorias[obligatorias > 0].items():
            if col not in ("partido_id", "partido_nombre", "partido_alias"):  # ver R11
                salida.append(Hallazgo(f"R14_nulos_{col}", "error",
                                       f"{n} nulos en columna obligatoria {col}.",
                                       n_afectados=int(n)))
        if not salida:
            salida.append(Hallazgo("R14_nulos", "ok", "Sin valores nulos."))
        return salida

    def _explicar_nulo(self, col: str) -> str | None:
        """Solo explica patrones verificados en los datos; si no calzan, devuelve None."""
        nulo = self.t[col].isna()
        if col == "articulo":
            tipos = set(self.t.loc[nulo, "tipo_votacion_proyecto_ley"])
            if tipos <= {"General"}:
                return "Votación en general: no se vota un artículo específico."
        if col == "fecha_termino_periodo" and nulo.all():
            return "Período legislativo vigente en la fuente: no tiene fecha de término."
        return None

    def _anomalias_f3(self) -> list[Hallazgo]:
        if self.calidad is None:
            return []
        rep = self.calidad[self.calidad["diputado_id"].isin(set(self.t["diputado_id"]))]
        salida = []
        for (tabla, tipo), grupo in rep.groupby(["tabla", "tipo"]):
            reglas = sorted(grupo["regla_aplicada"].dropna().unique())
            salida.append(Hallazgo(
                f"R15_f3_{tabla}_{tipo}", "advertencia",
                f"Anomalía ya registrada en reporte_calidad de F3 ({grupo.iloc[0].descripcion})",
                "Documentada en F3 con regla: " + " | ".join(reglas),
                n_afectados=grupo["diputado_id"].nunique(),
                ejemplos=tuple(sorted(grupo["diputado_id"].unique())[:MAX_EJEMPLOS])))
        return salida

    # ------------------------------------------------------------------------ perfil
    def perfil(self) -> dict:
        """Descriptivos para definir reglas y universo de análisis; no son hallazgos."""
        v = (self.t.groupby(["votacion_id", "opcion_voto"]).size().unstack(fill_value=0)
             .reindex(columns=list(CODIGOS_VOTO.values()), fill_value=0))
        meta = self.t.groupby("votacion_id")[["fecha", "tipo_votacion_proyecto_ley"]].first()
        votaciones = v.join(meta).assign(
            n=v.sum(axis=1), unanime=(v > 0).sum(axis=1) == 1).reset_index()
        partidos = (self.t.groupby(["partido_alias"]).agg(
            filas=("diputado_id", "size"), diputados=("diputado_id", "nunique"),
            votaciones=("votacion_id", "nunique"))
            .sort_values(["filas", "partido_alias"], ascending=[False, True]).reset_index())
        return {
            "votos_por_categoria": self.t["opcion_voto"].value_counts().to_dict(),
            "diputados": int(self.t["diputado_id"].nunique()),
            "partidos": int(self.t["partido_alias"].nunique()),
            "votaciones": votaciones.sort_values("fecha").astype(
                {"n": int, "unanime": bool}).to_dict("records"),
            "partidos_detalle": partidos.to_dict("records"),
        }

    # ------------------------------------------------------------------- lectura F3
    @classmethod
    def desde_config(cls, raiz: Path | str, config: Path | str) -> tuple["AuditorEntrada", dict]:
        """Lee rutas de analisis.toml [entrada] y las tablas F3 hermanas del corte."""
        raiz = Path(raiz)
        with (raiz / config).open("rb") as archivo:
            cfg = tomllib.load(archivo)["entrada"]
        ruta = raiz / cfg["ruta"]
        base = ruta.parent

        def leer(nombre: str) -> pd.DataFrame | None:
            archivo = base / nombre
            return pd.read_csv(archivo, dtype="string") if archivo.is_file() else None

        manifiesto_ruta = raiz / cfg["manifiesto_corte"]
        manifiesto = None
        if manifiesto_ruta.is_file() and manifiesto_ruta.stat().st_size:
            manifiesto = json.loads(manifiesto_ruta.read_text(encoding="utf-8"))
        fuentes = ["detalle_votaciones_procesado.csv", "proyecto_ley_procesado.csv",
                   "militancias_analiticas.csv", "diputados_procesados.csv",
                   "reporte_calidad.csv"]
        huella = calcular_sha256(ruta)
        entrada = {
            "entrada": {"ruta": cfg["ruta"], "sha256": huella},
            "manifiesto_corte": {
                "ruta": cfg["manifiesto_corte"],
                "commit_git": manifiesto.get("reproducibilidad", {}).get("commit_git"),
            } if manifiesto else None,
            "fuentes_contraste": {
                f: {"ruta": (base / f).relative_to(raiz).as_posix(),
                    "sha256": calcular_sha256(base / f)}
                for f in fuentes if (base / f).is_file()
            },
        }
        auditor = cls(
            pd.read_csv(ruta, dtype="string"),
            esperado={"filas": cfg["filas_esperadas"], "columnas": cfg["columnas_esperadas"],
                      "votaciones": cfg["votaciones_esperadas"]},
            detalle=leer(fuentes[0]), proyecto=leer(fuentes[1]), militancias=leer(fuentes[2]),
            padron=leer(fuentes[3]), calidad_f3=leer(fuentes[4]),
            manifiesto=manifiesto, sha256_tabla=huella,
        )
        return auditor, entrada


def guardar_json(reporte: ReporteAuditoria, destino: Path | str) -> Path:
    """Escritura determinista: mismo corte, mismo archivo byte a byte."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(reporte.a_dict(), ensure_ascii=False, indent=2, default=str)
    destino.write_text(texto + "\n", encoding="utf-8")
    return destino


RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]


def main(
    raiz: Path | str = RAIZ_REPOSITORIO, config: str = "F4/config/analisis.toml"
) -> ReporteAuditoria:
    """Ejecuta la auditoría y escribe F4/data/reports/auditoria_entrada.json."""
    auditor, entrada = AuditorEntrada.desde_config(raiz, config)
    reporte = auditor.auditar(entrada)
    destino = guardar_json(reporte, Path(raiz) / "F4/data/reports/auditoria_entrada.json")
    resumen = reporte.a_dict()["resumen"]
    relativo = destino.relative_to(Path(raiz).resolve()) if destino.is_absolute() else destino
    print(f"Entrada {'aprobada' if reporte.aprobado else 'bloqueada'} · {resumen} → {relativo}")
    return reporte


if __name__ == "__main__":
    main()
