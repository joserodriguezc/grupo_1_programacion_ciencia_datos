"""QA de matrices: universo por método.

Registra cuántos diputados, partidos y votaciones entran a cada análisis y por qué quedan
fuera los demás. Lee las matrices procesadas y las salidas ya generadas por cada método;
no recalcula modelos. La cobertura a 40, 60 y 80 % se informa para mostrar cómo cambiaría
cada universo, pero solo el clustering la usa como filtro (0,80): sin padrón verificable
el umbral principal no se aplica.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from F4.src.analisis.contratos import SALIDAS

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]
SALIDA = SALIDAS["universo_por_metodo"].ruta
UMBRALES = (0.40, 0.60, 0.80)
GRUPOS_NO_PARTIDARIOS = ("IND",)
MAX_INTEGRANTES_PARTIDO_PEQUENO = 2
SIN_DENOMINADOR = "no aplica: sin padrón no hay denominador de integrantes elegibles"

# Rutas desde el contrato de datos (contratos.py).
ENTRADAS = {nombre: SALIDAS[nombre].ruta for nombre in (
    "ternaria", "afiliacion", "bcall_seleccion", "bcall_diputados", "bcall_votaciones",
    "externo_seleccion", "posicion", "cohesion", "resumen_cohesion", "afinidad", "clusters",
    "exclusiones_clustering", "votaciones_clustering", "pca", "exclusiones_pca",
    "votaciones_pca")}

COLUMNAS = [
    "metodo", "rol", "vista", "unidad", "n_corpus", "n_incluidos", "n_excluidos",
    "motivos_exclusion", "n_diputados", "n_partidos", "n_votaciones_corpus", "n_votaciones",
    "votaciones_excluidas", "motivo_votaciones_excluidas", "orientacion_requerida",
    "n_votaciones_orientables", "unidad_cobertura",
    *[f"n_cobertura_{int(u * 100):03d}" for u in UMBRALES],
    "regla_inclusion", "excepciones", "estado", "fuente",
]


class ErrorUniverso(ValueError):
    """Las salidas de los métodos no permiten reconstruir su universo."""


# ----------------------------------------------------------------------------- utilidades
def _id(serie: pd.Series) -> pd.Series:
    """Normaliza identificadores (20629, 20629.0 y "20629" son la misma votación)."""
    numeros = pd.to_numeric(serie, errors="coerce")
    return numeros.astype("Int64").astype("string").where(numeros.notna(),
                                                          serie.astype("string"))


def motivos(razones: pd.Series) -> str | None:
    """'MOTIVO:n|MOTIVO:n' ordenado por frecuencia; None si no hay exclusiones."""
    conteo = razones.dropna().astype(str).value_counts()
    if conteo.empty:
        return None
    return "|".join(f"{motivo}:{n}" for motivo, n in conteo.items())


def contar_cobertura(cobertura: pd.Series, umbrales: Iterable[float] = UMBRALES) -> dict:
    """Unidades con cobertura >= cada umbral (la tolerancia evita errores de coma flotante)."""
    return {f"n_cobertura_{int(u * 100):03d}": int((cobertura >= u - 1e-12).sum())
            for u in umbrales}


def votaciones_informativas(ternaria: pd.DataFrame) -> list[str]:
    return [str(c) for c in ternaria.columns[ternaria.nunique(dropna=True).gt(1)]]


def cobertura_diputados(ternaria: pd.DataFrame) -> pd.Series:
    """Decisiones sustantivas sobre las votaciones informativas del corpus."""
    informativas = ternaria.loc[:, ternaria.nunique(dropna=True).gt(1)]
    if informativas.shape[1] == 0:
        raise ErrorUniverso("No hay votaciones informativas en la matriz ternaria.")
    return informativas.notna().sum(axis=1) / informativas.shape[1]


def partidos_pequenos(afiliacion: pd.DataFrame,
                      maximo: int = MAX_INTEGRANTES_PARTIDO_PEQUENO) -> pd.Series:
    """Partidos (sin grupos no partidarios) con a lo sumo `maximo` integrantes distintos."""
    tamanos = afiliacion.groupby("partido_id")["diputado_id"].nunique()
    tamanos = tamanos.drop(index=list(GRUPOS_NO_PARTIDARIOS), errors="ignore")
    return tamanos[tamanos <= maximo].sort_values()


def _texto_pequenos(pequenos: pd.Series, presentes: Iterable[str]) -> str | None:
    presentes = set(presentes)
    lista = [f"{p}({n})" for p, n in pequenos.items() if p in presentes]
    return f"Partidos con <= {MAX_INTEGRANTES_PARTIDO_PEQUENO} integrantes: " + ", ".join(
        lista) if lista else None


def _unir(*textos: str | None) -> str | None:
    partes = [t for t in textos if t]
    return " ".join(partes) if partes else None


# -------------------------------------------------------------------------------- modelo
class UniversoPorMetodo:
    """Una fila por método con su universo, exclusiones y coberturas alternativas."""

    def __init__(self, tablas: dict[str, pd.DataFrame], umbrales=UMBRALES) -> None:
        faltan = sorted(set(ENTRADAS) - set(tablas))
        if faltan:
            raise ErrorUniverso(f"Faltan tablas: {faltan}.")
        self.t = {nombre: tabla.copy() for nombre, tabla in tablas.items()}
        self.umbrales = tuple(umbrales)

        ternaria = self.t["ternaria"]
        if "diputado_id" in ternaria.columns:
            ternaria = ternaria.set_index("diputado_id")
        ternaria.index = _id(ternaria.index.to_series()).to_numpy()
        ternaria.columns = [str(c) for c in ternaria.columns]
        self.ternaria = ternaria
        self.votaciones_corpus = list(ternaria.columns)
        self.informativas = votaciones_informativas(ternaria)
        self.cobertura = cobertura_diputados(ternaria)

        afi = self.t["afiliacion"].copy()
        afi["diputado_id"], afi["votacion_id"] = _id(afi["diputado_id"]), _id(afi["votacion_id"])
        self.afiliacion = afi
        self.pequenos = partidos_pequenos(afi)

    # -------------------------------------------------------------------- auxiliares
    def _partidos_de(self, diputados: Iterable[str]) -> set[str]:
        afi = self.afiliacion
        partidos = set(afi.loc[afi["diputado_id"].isin(set(diputados)), "partido_id"])
        return partidos - set(GRUPOS_NO_PARTIDARIOS)

    def _diputados_en_celdas(self, celdas: pd.DataFrame,
                             diputados: Iterable[str] | None = None) -> set[str]:
        """Diputados con voto en alguna celda partido × votación incluida."""
        afi = self.afiliacion
        observado = self.ternaria.stack().dropna().index
        votos = pd.DataFrame(list(observado), columns=["diputado_id", "votacion_id"])
        votos = votos.merge(afi, on=["diputado_id", "votacion_id"], how="inner")
        if diputados is not None:
            votos = votos[votos["diputado_id"].isin(set(diputados))]
        claves = set(zip(celdas["partido_id"], _id(celdas["votacion_id"])))
        dentro = [c in claves for c in zip(votos["partido_id"], votos["votacion_id"])]
        return set(votos.loc[dentro, "diputado_id"])

    def _fila(self, **campos) -> dict:
        fila = dict.fromkeys(COLUMNAS)
        fila.update(campos)
        fila["n_excluidos"] = fila["n_corpus"] - fila["n_incluidos"]
        return fila

    def _cobertura_corpus(self) -> dict:
        return contar_cobertura(self.cobertura, self.umbrales)

    # ---------------------------------------------------------------------- métodos
    def bcall(self) -> list[dict]:
        sel = self.t["bcall_seleccion"].assign(diputado_id=lambda d: _id(d["diputado_id"]))
        dip = self.t["bcall_diputados"]
        vot = self.t["bcall_votaciones"].assign(votacion_id=lambda d: _id(d["votacion_id"]))
        usables = (vot["incluida_por_varianza"].astype(bool)
                   & vot["orientacion_definida"].astype(bool))
        incluidos = set(sel.loc[sel["incluido"].astype(bool), "diputado_id"])
        n_d2 = int(dip["d2"].isna().sum())
        externo = self.t["externo_seleccion"].assign(diputado_id=lambda d: _id(d["diputado_id"]))
        incluidos_ext = set(externo.loc[externo["incluido"].astype(bool), "diputado_id"])
        comun = dict(
            vista="ternaria", unidad="diputado", n_corpus=len(sel),
            n_votaciones_corpus=len(vot), n_votaciones=int(usables.sum()),
            votaciones_excluidas="|".join(vot.loc[~usables, "votacion_id"]),
            motivo_votaciones_excluidas=motivos(vot.loc[~usables, "razones_exclusion"]
                                                .astype(str).str.strip("[]\"'")),
            orientacion_requerida=True,
            n_votaciones_orientables=int(usables.sum()),
            unidad_cobertura="diputados del corpus con decisiones sobre votaciones informativas",
            **self._cobertura_corpus(),
        )
        return [
            self._fila(
                metodo="bcall_automatico", rol="principal",
                n_incluidos=len(incluidos), motivos_exclusion=motivos(sel["razon_exclusion"]),
                n_diputados=len(incluidos), n_partidos=len(self._partidos_de(incluidos)),
                regla_inclusion="participación > threshold sobre todas las votaciones "
                                "(>= 2 decisiones con 15 votaciones); sin cobertura principal",
                excepciones=_unir(
                    f"d2 NA para {n_d2} diputado(s) con menos de dos votaciones utilizables."
                    if n_d2 else None,
                    "Los excluidos solo registran la votación 42724.",
                ),
                estado=str(dip["estado"].dropna().iloc[0]) if "estado" in dip else None,
                fuente=ENTRADAS["bcall_diputados"], **comun,
            ),
            self._fila(
                metodo="bcall_grupos_externos", rol="complementario",
                n_incluidos=len(incluidos_ext),
                motivos_exclusion=motivos(externo["razon_exclusion"]),
                n_diputados=len(incluidos_ext), n_partidos=len(self._partidos_de(incluidos_ext)),
                regla_inclusion="grupo externo L/R único desde partidos observados y "
                                "participación > threshold",
                excepciones="SIN_CLUSTER corresponde a diputados sin clasificación externa "
                            "(independientes sin partido observado).",
                estado="DESCRIPTIVO_COMPLEMENTARIO",
                fuente=ENTRADAS["externo_seleccion"], **comun,
            ),
        ]

    def posicion(self) -> list[dict]:
        pos = self.t["posicion"]
        pv = pos[pos["nivel"].eq("partido_votacion")].copy()
        pp = pos[pos["nivel"].eq("partido")].copy()
        incluido = pv["incluido"].astype(bool)
        bcall_incluidos = set(_id(self.t["bcall_seleccion"].loc[
            self.t["bcall_seleccion"]["incluido"].astype(bool), "diputado_id"]))
        dip_celdas = self._diputados_en_celdas(pv[incluido], bcall_incluidos)
        partidos_pv = set(pv["partido_id"])
        no_partido = sorted(partidos_pv & set(GRUPOS_NO_PARTIDARIOS))
        aviso_ind = (f"{', '.join(no_partido)} no es partido: su P_p se calcula, pero no se "
                     "publica, como en cohesión." if no_partido else None)
        votaciones = sorted(set(_id(pv["votacion_id"])))
        cobertura_partido = pv.groupby("partido_id")["incluido"].mean()
        # Publicable: P_p estimable y grupo partidario (IND se calcula, pero no se publica).
        p_p = pp["publicable"].astype("string").str.lower().eq("true")
        comun = dict(rol="principal", vista="ternaria_estandarizada_y_orientada",
                     n_votaciones_corpus=len(self.votaciones_corpus), n_votaciones=len(votaciones),
                     votaciones_excluidas="|".join(
                         sorted(set(self.votaciones_corpus) - set(votaciones))),
                     motivo_votaciones_excluidas="SIN_VARIANZA_EN_BCALL",
                     orientacion_requerida=True, n_votaciones_orientables=len(votaciones),
                     estado="DESCRIPTIVO_SIN_PADRON", fuente=ENTRADAS["posicion"])
        return [
            self._fila(
                metodo="posicion_partido_votacion", unidad="partido_x_votacion",
                n_corpus=len(pv), n_incluidos=int(incluido.sum()),
                motivos_exclusion=motivos(pv["razon_exclusion"]),
                n_diputados=len(dip_celdas),
                n_partidos=len(set(pv.loc[incluido, "partido_id"]) - set(GRUPOS_NO_PARTIDARIOS)),
                unidad_cobertura=SIN_DENOMINADOR,
                regla_inclusion=">= min_decisiones_por_partido_votacion (2) decisiones en la celda",
                excepciones=_unir(aviso_ind, _texto_pequenos(self.pequenos, partidos_pv)),
                **comun,
            ),
            self._fila(
                metodo="posicion_partidaria_P_p", unidad="partido",
                n_corpus=len(pp), n_incluidos=int(p_p.sum()),
                motivos_exclusion=motivos(pp["razon_no_publicable"]),
                n_diputados=len(self._diputados_en_celdas(
                    pv[incluido & pv["partido_id"].isin(pp.loc[p_p, "partido_id"])],
                    bcall_incluidos)),
                n_partidos=len(set(pp.loc[p_p, "partido_id"]) - set(GRUPOS_NO_PARTIDARIOS)),
                unidad_cobertura=("partidos con celdas incluidas sobre sus votaciones "
                                  "con integrantes"),
                **contar_cobertura(cobertura_partido, self.umbrales),
                regla_inclusion=">= min_votaciones_publicacion (2) votaciones con celda incluida",
                excepciones=_unir(aviso_ind, _texto_pequenos(self.pequenos, partidos_pv)),
                **comun,
            ),
        ]

    def cohesion(self) -> list[dict]:
        c = self.t["cohesion"].copy()
        r = self.t["resumen_cohesion"].copy()
        votaciones = sorted(set(_id(c["votacion_id"])))
        partidos = set(c["partido_id"])
        comun = dict(rol="principal", vista="nominal",
                     n_votaciones_corpus=len(self.votaciones_corpus), n_votaciones=len(votaciones),
                     votaciones_excluidas=None, orientacion_requerida=False,
                     estado="DESCRIPTIVO_SIN_PADRON", fuente=ENTRADAS["cohesion"])
        aviso_unanimes = ("Las votaciones unánimes se conservan y se marcan; el resumen también "
                          "informa la mediana sin ellas.")
        filas = []
        for metodo, columna, regla in (
            ("cohesion_ai_entropia", "publicable_ai_entropia",
             "T >= min_decisiones_publicacion_partido_votacion (2); IND no se publica"),
            ("cohesion_rice", "publicable_rice",
             "Y + N >= min_votos_binarios_publicacion_rice (2); IND no se publica"),
        ):
            publica = c[columna].astype(bool)
            filas.append(self._fila(
                metodo=metodo, unidad="partido_x_votacion", n_corpus=len(c),
                n_incluidos=int(publica.sum()),
                motivos_exclusion=motivos(c.loc[~publica, "razon_no_publicable"]),
                n_diputados=len(self._diputados_en_celdas(c[publica])),
                n_partidos=len(set(c.loc[publica, "partido_id"]) - set(GRUPOS_NO_PARTIDARIOS)),
                unidad_cobertura=SIN_DENOMINADOR,
                regla_inclusion=regla,
                excepciones=_unir(aviso_unanimes, _texto_pequenos(self.pequenos, partidos)),
                **comun,
            ))
        publica = r["publicable_resumen"].astype(bool)
        cobertura = (pd.to_numeric(r["n_validas_ai"]) / pd.to_numeric(r["n_votaciones_observadas"]))
        filas.append(self._fila(
            metodo="cohesion_resumen_partido", unidad="partido", n_corpus=len(r),
            n_incluidos=int(publica.sum()),
            motivos_exclusion=motivos(r.loc[~publica, "razon_no_publicable_resumen"]),
            n_diputados=len(self._diputados_en_celdas(
                c[c["publicable_ai_entropia"].astype(bool)
                  & c["partido_id"].isin(r.loc[publica, "partido_id"])])),
            n_partidos=len(set(r.loc[publica, "partido_id"]) - set(GRUPOS_NO_PARTIDARIOS)),
            unidad_cobertura="grupos con votaciones válidas de AI sobre votaciones observadas",
            **contar_cobertura(cobertura.where(r["tipo_grupo"].eq("partido"), np.nan),
                               self.umbrales),
            regla_inclusion=">= min_votaciones_resumen_principal (2) votaciones válidas",
            excepciones=_unir("IND se calcula como grupo de independientes y no se publica.",
                              _texto_pequenos(self.pequenos, partidos)),
            **{**comun, "fuente": "F4/data/results/partidos/resumen_cohesion.csv"},
        ))
        return filas

    def afinidad(self) -> list[dict]:
        a = self.t["afinidad"]
        incluido = a["incluido"].astype(bool)
        diputados = (set(_id(a.loc[incluido, "diputado_i"]))
                     | set(_id(a.loc[incluido, "diputado_j"])))
        return [self._fila(
            metodo="afinidad_pares", rol="auxiliar", vista="nominal",
            unidad="par_de_diputados", n_corpus=len(a), n_incluidos=int(incluido.sum()),
            motivos_exclusion=motivos(a.loc[~incluido, "razon_exclusion"]),
            n_diputados=len(diputados), n_partidos=len(self._partidos_de(diputados)),
            n_votaciones_corpus=len(self.votaciones_corpus),
            n_votaciones=len(self.votaciones_corpus), orientacion_requerida=False,
            unidad_cobertura="pares con co-votos sobre votaciones del corpus",
            **contar_cobertura(pd.to_numeric(a["cobertura_corpus"]), self.umbrales),
            regla_inclusion=">= pares.min_covotos (2) co-votos",
            excepciones="Acuerdo y Hamming se informan también sin votaciones unánimes.",
            estado="DESCRIPTIVO_SIN_PADRON", fuente=ENTRADAS["afinidad"],
        )]

    def _metodo_contraste(self, metodo, incluidos, exclusiones, votaciones, regla, excepciones,
                    estado, fuente, vista) -> dict:
        votaciones = votaciones.assign(votacion_id=lambda d: _id(d["votacion_id"]))
        dentro = votaciones["incluida"].astype(bool)
        ids = set(_id(incluidos["diputado_id"]))
        return self._fila(
            metodo=metodo, rol="contraste_descriptivo", vista=vista,
            unidad="diputado", n_corpus=len(ids) + len(exclusiones), n_incluidos=len(ids),
            motivos_exclusion=motivos(exclusiones["razon_exclusion"]),
            n_diputados=len(ids), n_partidos=len(self._partidos_de(ids)),
            n_votaciones_corpus=len(votaciones), n_votaciones=int(dentro.sum()),
            votaciones_excluidas="|".join(votaciones.loc[~dentro, "votacion_id"]),
            motivo_votaciones_excluidas=motivos(votaciones.loc[~dentro, "razon"]),
            orientacion_requerida=False,
            unidad_cobertura="diputados del corpus con decisiones sobre votaciones informativas",
            **self._cobertura_corpus(), regla_inclusion=regla, excepciones=excepciones,
            estado=estado, fuente=fuente,
        )

    def contrastes(self) -> list[dict]:
        clusters = self.t["clusters"]
        cobertura_minima = float(clusters["cobertura_minima"].iloc[0])
        return [
            self._metodo_contraste(
                "clustering", clusters, self.t["exclusiones_clustering"],
                self.t["votaciones_clustering"],
                f"cobertura >= {cobertura_minima:.2f} sobre votaciones informativas y todos los "
                "pares con co-votos suficientes; k = 2",
                "Sensible al bloque articulos_206xx (ARI = 0,461).",
                str(clusters["estado"].iloc[0]), ENTRADAS["clusters"], "nominal"),
            self._metodo_contraste(
                "pca", self.t["pca"], self.t["exclusiones_pca"], self.t["votaciones_pca"],
                "casos completos en todas las votaciones informativas; sin imputación",
                "El universo de casos completos es menor que el de B-Call.",
                str(self.t["pca"]["estado"].iloc[0]), ENTRADAS["pca"], "ternaria"),
        ]

    def calcular(self) -> pd.DataFrame:
        filas = (self.bcall() + self.posicion() + self.cohesion() + self.afinidad()
                 + self.contrastes())
        tabla = pd.DataFrame(filas, columns=COLUMNAS)
        conteos = [c for c in COLUMNAS if c.startswith("n_")]
        tabla[conteos] = tabla[conteos].astype("Int64")
        if (tabla["n_incluidos"] + tabla["n_excluidos"] != tabla["n_corpus"]).any():
            raise ErrorUniverso("Incluidos y excluidos no suman el corpus de algún método.")
        return tabla


# ------------------------------------------------------------------------- ejecución
def desde_repositorio(raiz: Path | str = RAIZ_REPOSITORIO) -> UniversoPorMetodo:
    raiz = Path(raiz)
    tablas = {}
    for nombre, ruta in ENTRADAS.items():
        archivo = raiz / ruta
        if not archivo.is_file() or archivo.stat().st_size == 0:
            raise ErrorUniverso(f"Falta la salida {nombre}: {ruta}. Ejecute antes ese método.")
        tablas[nombre] = pd.read_csv(archivo, dtype={"partido_id": "string"})
    return UniversoPorMetodo(tablas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", default=SALIDA)
    args = parser.parse_args(argv)
    tabla = desde_repositorio().calcular()
    salida = RAIZ_REPOSITORIO / args.salida
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False, lineterminator="\n")
    columnas = ["metodo", "unidad", "n_corpus", "n_incluidos", "n_excluidos", "n_votaciones"]
    print(tabla[columnas].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
