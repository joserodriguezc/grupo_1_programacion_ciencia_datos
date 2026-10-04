"""Conciliación cruzada de resultados entre unidades, codificaciones y métodos.

Compara los conteos y universos de cada salida con la fuente y entre sí. No recalcula
ningún método: lee los CSV y JSON ya generados. Cada control termina como "coincide"
(mismo número), "explicada" (diferencia con causa, universo y decisión documentados) o
"pendiente" (diferencia sin causa conocida o salida faltante). Ninguna diferencia se
corrige aquí.
"""

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from F4.src.analisis.corte import calcular_sha256

Estado = Literal["coincide", "explicada", "pendiente"]
RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]
SALIDA = "F4/data/reports/conciliacion_resultados.json"
ENTRADAS = {
    "votos": "F4/data/processed/votos_codificados.csv",
    "afiliacion": "F4/data/processed/afiliacion_por_votacion.csv",
    "mascara": "F4/data/processed/mascara_observacion.csv",
    "binaria": "F4/data/processed/matriz_binaria.csv",
    "ternaria": "F4/data/processed/matriz_ternaria.csv",
    "participacion": "F4/data/reports/participacion_observada.csv",
    "conteos_participacion": "F4/data/reports/conteos_partido_votacion.csv",
    "cohesion": "F4/data/results/partidos/cohesion_por_votacion.csv",
    "resumen_cohesion": "F4/data/results/partidos/resumen_cohesion.csv",
    "posicion": "F4/data/results/partidos/posicion_partidaria.csv",
    "afinidad": "F4/data/results/pares/afinidad_diputados.csv",
    "hamming_partidos": "F4/data/results/pares/hamming_partidos.csv",
    "sensibilidad": "F4/data/reports/sensibilidad.csv",
    "bcall_diputados": "F4/data/results/individual/bcall/bcall_diputados.csv",
    "universo_por_metodo": "F4/data/reports/universo_por_metodo.csv",
}
JSON_ENTRADAS = {"auditoria": "F4/data/reports/auditoria_entrada.json"}


@dataclass(frozen=True, slots=True)
class Control:
    """Comparación entre dos conteos o universos, con su explicación."""

    control: str
    referencia: str
    comparado: str
    n_referencia: int | None
    n_comparado: int | None
    estado: Estado
    causa: str | None = None
    universo: str | None = None
    decision: str | None = None

    @property
    def diferencia(self) -> int | None:
        if self.n_referencia is None or self.n_comparado is None:
            return None
        return self.n_comparado - self.n_referencia

    def a_dict(self) -> dict:
        return asdict(self) | {"diferencia": self.diferencia}


def _control(nombre, ref, comp, n_ref, n_comp, *, causa=None, universo=None,
             decision=None) -> Control:
    """Coincide si los números son iguales; si no, explicada solo cuando hay causa."""
    if n_ref is not None and n_comp is not None and int(n_ref) == int(n_comp):
        return Control(nombre, ref, comp, int(n_ref), int(n_comp), "coincide",
                       universo=universo)
    estado = "explicada" if causa and decision else "pendiente"
    return Control(nombre, ref, comp, None if n_ref is None else int(n_ref),
                   None if n_comp is None else int(n_comp), estado, causa, universo, decision)


def _id(serie: pd.Series) -> pd.Series:
    """Normaliza IDs leídos como número o texto ('20629.0' -> '20629')."""
    return serie.astype("string").str.replace(r"\.0$", "", regex=True)


class ConciliadorResultados:
    """Aplica los controles cruzados sobre las salidas disponibles.

    tablas: DataFrames por nombre de ENTRADAS (None o vacío si la salida no existe).
    auditoria: contenido de auditoria_entrada.json.
    min_covotos, min_decisiones_partido_votacion: reglas de analisis.toml.
    tolerancia_posicion: diferencia |P_p − mediana d1| desde la que se revisa la
    heterogeneidad interna (en desviaciones estándar por votación).
    """

    def __init__(self, tablas: dict, auditoria: dict, *, min_covotos: int = 2,
                 min_decisiones_partido_votacion: int = 2,
                 tolerancia_posicion: float = 0.10) -> None:
        self.t = tablas
        self.auditoria = auditoria
        self.min_covotos = min_covotos
        self.min_dec_pv = min_decisiones_partido_votacion
        self.tolerancia = tolerancia_posicion
        v = tablas["votos"].copy()
        v["diputado_id"], v["votacion_id"] = _id(v["diputado_id"]), _id(v["votacion_id"])
        self.votos = v[v["observado"].astype("string").str.lower() == "true"]

    def disponible(self, nombre: str) -> bool:
        tabla = self.t.get(nombre)
        return tabla is not None and not tabla.empty

    # ---------------------------------------------------------------- conteos base
    def votaciones_unanimes(self) -> set[str]:
        n = self.votos.groupby("votacion_id")["voto_nominal"].nunique()
        return set(n[n == 1].index)

    def controles_fuente(self) -> list[Control]:
        n = len(self.votos)
        auditados = int(sum(self.auditoria["perfil"]["votos_por_categoria"].values()))
        mascara = self.t["mascara"].set_index("diputado_id").astype("boolean")
        part = self.t["participacion"]
        salida = [
            _control("decisiones_fuente_vs_auditoria", "auditoria_entrada",
                     "votos_codificados", auditados, n, universo="corte completo"),
            _control("decisiones_vs_mascara", "votos_codificados", "mascara_observacion",
                     n, int(mascara.sum().sum()), universo="corte completo"),
            _control("decisiones_vs_participacion", "votos_codificados",
                     "participacion_observada", n,
                     int(part["n_votos_observados"].astype(int).sum()),
                     universo="diputados con al menos un voto"),
            _control("celdas_sin_registro", "mascara_observacion",
                     "participacion_observada", int((~mascara).sum().sum()),
                     int(part["n_sin_registro"].astype(int).sum()),
                     universo="diputados con al menos un voto × votaciones"),
            _control("decisiones_vs_cohesion", "votos_codificados", "cohesion_por_votacion",
                     n, int(self.t["cohesion"]["T"].astype(int).sum()),
                     universo="todas las decisiones, incluidas unánimes e independientes"),
        ]
        abst = int((self.votos["voto_nominal"] == "Abstención").sum())
        binaria = self.t["binaria"].set_index("diputado_id")
        salida.append(_control(
            "vista_binaria", "votos_codificados", "matriz_binaria", n,
            int(binaria.notna().sum().sum()),
            causa=f"La vista binaria excluye las {abst} abstenciones.",
            universo="decisiones Sí/No",
            decision="Correcto por diseño: Rice usa solo Sí/No; AI y entropía usan las tres "
                     "categorías."))
        ternaria = self.t["ternaria"].set_index("diputado_id").apply(pd.to_numeric)
        salida.append(_control("abstenciones_en_ternaria", "votos_codificados",
                               "matriz_ternaria (valor 0)", abst, int((ternaria == 0).sum().sum()),
                               universo="abstenciones"))
        return salida

    # ---------------------------------------------------------- partido × votación
    def controles_partido_votacion(self) -> list[Control]:
        afil = self.t["afiliacion"].assign(diputado_id=lambda d: _id(d["diputado_id"]),
                                           votacion_id=lambda d: _id(d["votacion_id"]))
        ref = afil.groupby(["partido_id", "votacion_id"]).size()
        salida = []
        for nombre, tabla in (("cohesion", self.t["cohesion"]),
                              ("conteos_participacion", self.t["conteos_participacion"])):
            comp = (tabla.assign(votacion_id=_id(tabla["votacion_id"]))
                    .set_index(["partido_id", "votacion_id"])["T"].astype(int))
            unidas = pd.concat([ref.rename("ref"), comp.rename("comp")], axis=1).fillna(0)
            distintas = int((unidas["ref"] != unidas["comp"]).sum())
            salida.append(_control(
                f"celdas_partido_votacion_{nombre}", "afiliacion_por_votacion", nombre,
                0, distintas, universo=f"{len(unidas)} celdas partido × votación"))
        return salida

    # ------------------------------------------------------------- universo B-Call
    def descomposicion_bcall(self) -> dict:
        """Pasos desde todas las decisiones hasta el universo de la posición partidaria."""
        unanimes = self.votaciones_unanimes()
        part = self.t["participacion"].assign(diputado_id=lambda d: _id(d["diputado_id"]))
        bajo = set(part.loc[part["supera_umbral_bcall"].astype("string").str.lower()
                            == "false", "diputado_id"])
        paso1 = self.votos[~self.votos["votacion_id"].isin(unanimes)]
        paso2 = paso1[~paso1["diputado_id"].isin(bajo)]
        return {"n_total": len(self.votos), "votaciones_unanimes": sorted(unanimes),
                "n_en_unanimes": len(self.votos) - len(paso1),
                "diputados_bajo_filtro": len(bajo),
                "n_de_diputados_bajo_filtro": len(paso1) - len(paso2),
                "n_universo_esperado": len(paso2), "_tabla": paso2}

    def controles_posicion(self) -> list[Control]:
        if not self.disponible("posicion"):
            return [_control("universo_posicion", "votos_codificados",
                             "posicion_partidaria", len(self.votos), None)]
        d = self.descomposicion_bcall()
        pos = self.t["posicion"]
        pv = pos[pos["nivel"] == "partido_votacion"].assign(
            votacion_id=lambda x: _id(x["votacion_id"]),
            n_decisiones=lambda x: x["n_decisiones"].astype(int))
        salida = [_control(
            "universo_posicion", "votos_codificados", "posicion_partidaria (partido × votación)",
            d["n_total"], int(pv["n_decisiones"].sum()),
            causa=(f"B-Call excluye {d['n_en_unanimes']} decisiones de las votaciones unánimes "
                   f"{', '.join(d['votaciones_unanimes'])} (sin varianza) y "
                   f"{d['n_de_diputados_bajo_filtro']} decisiones de "
                   f"{d['diputados_bajo_filtro']} diputados que no superan el filtro de "
                   f"participación."),
            universo="votaciones con varianza × diputados sobre el filtro",
            decision="Diferencia esperada por diseño de B-Call; la cohesión conserva estas "
                     "decisiones." if d["n_universo_esperado"] == int(pv["n_decisiones"].sum())
            else None)]
        esperado = d["_tabla"].groupby(["partido_id", "votacion_id"]).size()
        obs = pv.set_index(["partido_id", "votacion_id"])["n_decisiones"]
        unidas = pd.concat([esperado.rename("e"), obs.rename("o")], axis=1).fillna(0)
        salida.append(_control("celdas_universo_posicion", "universo B-Call esperado",
                               "posicion_partidaria", 0, int((unidas["e"] != unidas["o"]).sum()),
                               universo=f"{len(unidas)} celdas partido × votación"))
        pp = pos[pos["nivel"] == "partido"].set_index("partido_id")
        incluidas = (pv[pv["incluido"].astype("string").str.lower() == "true"]
                     .groupby("partido_id")["n_decisiones"].sum())
        agregado = pp["n_decisiones"].astype(int)
        distintos = int((incluidas.reindex(agregado.index, fill_value=0) != agregado).sum())
        n_excl = int((pv["n_decisiones"] < self.min_dec_pv).sum())
        salida.append(_control(
            "posicion_nivel_partido", "suma de celdas incluidas", "posicion_partidaria (partido)",
            0, distintos, universo=f"{len(agregado)} partidos"))
        salida.append(Control(
            "celdas_bajo_minimo_posicion", "posicion_partidaria (partido × votación)",
            "posicion_partidaria (partido)", None, n_excl, "explicada",
            f"{n_excl} celdas partido × votación con menos de {self.min_dec_pv} decisiones "
            "no entran al perfil del partido.", "celdas partido × votación",
            "Regla [posicion_partidos].min_decisiones_por_partido_votacion de analisis.toml."))
        return salida

    # ---------------------------------------------------------------------- pares
    def controles_afinidad(self) -> list[Control]:
        if not self.disponible("afinidad"):
            return [_control("pares_afinidad", "mascara_observacion", "afinidad_diputados",
                             None, None)]
        a = self.t["afinidad"]
        m = self.t["mascara"].set_index("diputado_id").astype(bool)
        m.index = _id(pd.Series(m.index)).values
        n = len(m)
        x = m.to_numpy(dtype=int)
        co = x @ x.T
        pos = {d: i for i, d in enumerate(m.index)}
        i = _id(a["diputado_i"]).map(pos).to_numpy()
        j = _id(a["diputado_j"]).map(pos).to_numpy()
        calc = co[i, j]
        suma = (a["acuerdo"] + a["hamming"]).dropna()
        incluido = a["incluido"].astype("string").str.lower() == "true"
        return [
            _control("pares_afinidad", "diputados de la máscara", "afinidad_diputados",
                     n * (n - 1) // 2, len(a), universo="pares de diputados"),
            _control("covotos_vs_mascara", "mascara_observacion", "afinidad_diputados", 0,
                     int((calc != a["n_covotos"].astype(int).to_numpy()).sum()),
                     universo="pares con co-votos recalculados"),
            _control("acuerdo_mas_hamming", "identidad acuerdo + Hamming = 1",
                     "afinidad_diputados", 0, int(((suma - 1).abs() > 1e-9).sum()),
                     universo="pares con co-votos"),
            _control("inclusion_pares", f"n_covotos >= {self.min_covotos}", "afinidad_diputados",
                     0, int(((a["n_covotos"].astype(int) >= self.min_covotos) != incluido).sum()),
                     universo="pares de diputados"),
        ]

    # ---------------------------------------------------------------- sensibilidad
    def controles_sensibilidad(self) -> list[Control]:
        """Los valores base de la sensibilidad deben ser los mismos resultados publicados."""
        if not self.disponible("sensibilidad"):
            return [_control("sensibilidad_disponible", "posicion_partidaria", "sensibilidad",
                             None, None)]
        s = self.t["sensibilidad"].assign(id=lambda d: _id(d["id"]))
        salida = []
        if self.disponible("posicion"):
            pos = self.t["posicion"]
            pp = pd.to_numeric(pos[pos["nivel"] == "partido"].set_index("partido_id")["P_p"],
                               errors="coerce")
            base = (s[s["metodo"] == "perfil_partidario_P_p"].groupby("id")["valor_base"]
                    .first().astype(float))
            unidas = pd.concat([pp.rename("pos"), base.rename("sens")], axis=1)
            distintas = int((((unidas["pos"] - unidas["sens"]).abs() > 1e-9)
                             | (unidas["pos"].isna() != unidas["sens"].isna())).sum())
            salida.append(_control("sensibilidad_base_P_p", "posicion_partidaria",
                                   "sensibilidad (valor_base)", 0, distintas,
                                   universo=f"{len(unidas)} partidos"))
        if self.disponible("hamming_partidos"):
            h = self.t["hamming_partidos"]
            publicada = pd.Series(h["hamming"].astype(float).to_numpy(),
                                  index=h["partido_a"] + "|" + h["partido_b"])
            base = (s[s["metodo"] == "hamming_entre_partidos"].groupby("id")["valor_base"]
                    .first().astype(float))
            unidas = pd.concat([publicada.rename("pub"), base.rename("sens")], axis=1)
            distintas = int((((unidas["pub"] - unidas["sens"]).abs() > 1e-9)
                             | (unidas["pub"].isna() != unidas["sens"].isna())).sum())
            salida.append(_control("sensibilidad_base_afinidad", "hamming_partidos",
                                   "sensibilidad (valor_base)", 0, distintas,
                                   universo=f"{len(unidas)} pares de partidos"))
        d = self.descomposicion_bcall()
        esperados = d["_tabla"]["diputado_id"].nunique()
        d1 = s[s["metodo"] == "bcall_d1"].groupby("id")["valor_base"].first()
        salida.append(_control("sensibilidad_diputados_d1", "universo B-Call esperado",
                               "sensibilidad (d1 base)", esperados, int(d1.notna().sum()),
                               universo="diputados sobre el filtro de participación"))
        return salida

    # ------------------------------------------------------- B-Call individual
    def controles_bcall_individual(self) -> list[Control]:
        """La tabla individual de B-Call debe cubrir el universo esperado y coincidir
        con los valores base de la sensibilidad."""
        if not self.disponible("bcall_diputados"):
            return []  # salidas_faltantes ya la informa como pendiente
        b = self.t["bcall_diputados"].assign(diputado_id=lambda d: _id(d["diputado_id"]))
        esperados = set(self.descomposicion_bcall()["_tabla"]["diputado_id"])
        salida = [
            _control("diputados_bcall", "universo B-Call esperado", "bcall_diputados",
                     len(esperados), len(b),
                     universo="diputados sobre el filtro de participación"),
            _control("diputados_bcall_identidad", "universo B-Call esperado",
                     "bcall_diputados", 0, len(esperados ^ set(b["diputado_id"])),
                     universo="diputados presentes en uno solo de los dos conjuntos"),
        ]
        if self.disponible("sensibilidad"):
            s = self.t["sensibilidad"].assign(id=lambda x: _id(x["id"]))
            base = (s[s["metodo"] == "bcall_d1"].groupby("id")["valor_base"].first()
                    .astype(float))
            d1 = pd.to_numeric(b.set_index("diputado_id")["d1"], errors="coerce")
            unidas = pd.concat([d1.rename("bcall"), base.rename("sens")], axis=1)
            distintas = int((((unidas["bcall"] - unidas["sens"]).abs() > 1e-9)
                             | (unidas["bcall"].isna() != unidas["sens"].isna())).sum())
            salida.append(_control("bcall_d1_vs_sensibilidad", "bcall_diputados (d1)",
                                   "sensibilidad (d1 base)", 0, distintas,
                                   universo=f"{len(unidas)} diputados"))
        return salida

    # ------------------------------------------------------------- casos de revisión
    def partidos_cambiantes(self) -> list[dict]:
        afil = self.t["afiliacion"].assign(diputado_id=lambda d: _id(d["diputado_id"]))
        por = afil.groupby("diputado_id")["partido_alias"].agg(
            lambda s: sorted(s.dropna().unique()))
        cambian = por[por.map(len) > 1]
        return [{"diputado_id": d, "partidos": p} for d, p in cambian.items()]

    def posicion_vs_d1(self) -> list[dict]:
        """P_p frente a la mediana de d1 de sus integrantes, con la cohesión como contexto."""
        if not self.disponible("posicion"):
            return []
        pos = self.t["posicion"]
        pp = pos[pos["nivel"] == "partido"].set_index("partido_id")
        res = self.t["resumen_cohesion"].set_index("partido_id")
        filas = []
        for pid, f in pp.iterrows():
            p_p, med, iqr = (pd.to_numeric(f[c], errors="coerce")
                             for c in ("P_p", "mediana_d1", "iqr_d1"))
            ai = pd.to_numeric(res["mediana_ai"].get(pid), errors="coerce")
            if pd.isna(p_p):
                clase = "sin_P_p"
            else:
                dif = abs(p_p - med)
                if dif <= self.tolerancia:
                    clase = "consistente"
                elif iqr >= dif or (pd.notna(ai) and ai < 1):
                    clase = "divergente_explicada_por_heterogeneidad"
                else:
                    clase = "divergente_pendiente"
            filas.append({"partido_id": pid, "P_p": _f(p_p), "mediana_d1": _f(med),
                          "iqr_d1": _f(iqr), "diferencia": _f(p_p - med),
                          "mediana_ai": _f(ai), "clase": clase})
        return sorted(filas, key=lambda r: r["partido_id"])

    # ---------------------------------------------------------------- excepciones
    def salidas_faltantes(self) -> list[Control]:
        faltan = []
        for nombre, descripcion in (
            ("bcall_diputados", "la salida individual de B-Call (d1, d2, m_i por diputado); "
                                "sus valores base solo existen dentro de sensibilidad.csv"),
            ("universo_por_metodo", "el universo autorizado por método"),
            ("hamming_partidos", "la distancia de Hamming entre partidos (afinidad.py)"),
        ):
            if not self.disponible(nombre):
                faltan.append(Control(
                    f"salida_{nombre}", "salidas esperadas", ENTRADAS[nombre], None, None,
                    "pendiente", f"Archivo vacío: falta {descripcion}.", None,
                    "Sin decisión: debe generarla quien implementa el método."))
        return faltan

    def conciliar(self, trazabilidad: dict | None = None) -> dict:
        controles = (self.controles_fuente() + self.controles_partido_votacion()
                     + self.controles_posicion() + self.controles_afinidad()
                     + self.controles_sensibilidad() + self.controles_bcall_individual()
                     + self.salidas_faltantes())
        pvd = self.posicion_vs_d1()
        for fila in pvd:
            if fila["clase"] == "divergente_pendiente":
                controles.append(Control(
                    f"posicion_vs_d1_{fila['partido_id']}", "mediana_d1", "P_p", None, None,
                    "pendiente", "P_p se aleja de la mediana de d1 sin heterogeneidad ni "
                    "división interna que lo explique."))
        conteo = {e: sum(c.estado == e for c in controles)
                  for e in ("coincide", "explicada", "pendiente")}
        d = self.descomposicion_bcall()
        return {
            "reporte": "conciliacion_resultados",
            "trazabilidad": trazabilidad or {},
            "resumen": conteo,
            "controles": [c.a_dict() for c in controles],
            "descomposicion_universo_bcall": {k: v for k, v in d.items()
                                              if not k.startswith("_")},
            "votaciones_unanimes": {
                "ids": d["votaciones_unanimes"],
                "tratamiento": {
                    "cohesion": "se conservan y se marcan (votacion_unanime_sala)",
                    "posicion_partidaria": "excluidas por B-Call (sin varianza)",
                    "afinidad": "se informan columnas con y sin unánimes",
                }},
            "partidos_cambiantes": {
                "n_diputados": len(cambian := self.partidos_cambiantes()),
                "tratamiento": "cohesión y posición usan el partido vigente en cada votación; "
                               "ver controles celdas_partido_votacion_* y "
                               "celdas_universo_posicion",
                "detalle": cambian},
            "posicion_vs_d1": {"tolerancia": self.tolerancia, "detalle": pvd},
            "excepciones": [c.a_dict() for c in controles if c.estado != "coincide"],
        }


def _f(valor) -> float | None:
    return None if valor is None or pd.isna(valor) else round(float(valor), 6)


def _leer(ruta: Path) -> pd.DataFrame | None:
    if not ruta.is_file() or ruta.stat().st_size == 0:
        return None
    return pd.read_csv(ruta, dtype={"diputado_id": "string", "diputado_i": "string",
                                    "diputado_j": "string", "votacion_id": "string",
                                    "id": "string"})


def desde_repositorio(raiz: Path | str = RAIZ_REPOSITORIO) -> tuple[ConciliadorResultados,
                                                                     dict]:
    raiz = Path(raiz)
    tablas = {n: _leer(raiz / r) for n, r in ENTRADAS.items()}
    auditoria = json.loads((raiz / JSON_ENTRADAS["auditoria"]).read_text(encoding="utf-8"))
    with (raiz / "F4/config/analisis.toml").open("rb") as archivo:
        cfg = tomllib.load(archivo)
    trazabilidad = {r: (calcular_sha256(raiz / r) if (raiz / r).is_file()
                        and (raiz / r).stat().st_size else None)
                    for r in [*ENTRADAS.values(), *JSON_ENTRADAS.values()]}
    conciliador = ConciliadorResultados(
        tablas, auditoria, min_covotos=int(cfg["pares"]["min_covotos"]),
        min_decisiones_partido_votacion=int(
            cfg["posicion_partidos"]["min_decisiones_por_partido_votacion"]))
    return conciliador, trazabilidad


def main(raiz: Path | str = RAIZ_REPOSITORIO) -> dict:
    conciliador, trazabilidad = desde_repositorio(raiz)
    reporte = conciliador.conciliar(trazabilidad)
    destino = Path(raiz) / SALIDA
    destino.write_text(json.dumps(reporte, ensure_ascii=False, indent=2, default=str) + "\n",
                       encoding="utf-8", newline="\n")
    print(f"Conciliación: {reporte['resumen']} → {SALIDA}")
    return reporte


if __name__ == "__main__":
    main()
