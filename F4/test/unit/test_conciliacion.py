"""Conciliación cruzada: estados de los controles, universo de B-Call y casos de revisión."""

from pathlib import Path

import pandas as pd

from F4.src.analisis.conciliacion import ConciliadorResultados, _control, desde_repositorio

RAIZ = Path(__file__).resolve().parents[3]

# Ejemplo de referencia: C sin decisión en j4; B se abstiene en j4; j2 es unánime.
GUIA = {"A": ["No", "Sí", "No", "Sí"], "B": ["No", "Sí", "No", "Abstención"],
        "C": ["No", "Sí", "Sí", None], "D": ["Sí", "Sí", "Sí", "No"],
        "E": ["Sí", "Sí", "Sí", "No"]}
PARTIDO = {"A": "p1", "B": "p1", "C": "p1", "D": "p2", "E": "p2"}
VOTACIONES = ["j1", "j2", "j3", "j4"]


def votos_guia() -> pd.DataFrame:
    filas = [{"diputado_id": d, "votacion_id": j, "partido_id": PARTIDO[d],
              "voto_nominal": v, "observado": "True"}
             for d, votos in GUIA.items() for j, v in zip(VOTACIONES, votos) if v]
    return pd.DataFrame(filas, dtype="string")


def conciliador(**tablas) -> ConciliadorResultados:
    return ConciliadorResultados({"votos": votos_guia(), **tablas}, {})


def test_control_coincide_explicada_o_pendiente():
    assert _control("x", "a", "b", 5, 5).estado == "coincide"
    assert _control("x", "a", "b", 5, 4, causa="c", decision="d").estado == "explicada"
    assert _control("x", "a", "b", 5, 4, causa="c").estado == "pendiente"  # sin decisión
    assert _control("x", "a", "b", 5, None).estado == "pendiente"
    assert _control("x", "a", "b", 5, 4).diferencia == -1


def test_descomposicion_del_universo_bcall():
    part = pd.DataFrame({"diputado_id": list("ABCDE"),
                         "supera_umbral_bcall": ["True", "True", "False", "True", "True"]})
    d = conciliador(participacion=part).descomposicion_bcall()
    assert d["votaciones_unanimes"] == ["j2"]
    assert (d["n_total"], d["n_en_unanimes"]) == (19, 5)
    assert d["n_de_diputados_bajo_filtro"] == 2  # C en j1 y j3; j2 ya salió
    assert d["n_universo_esperado"] == 12


def posicion(filas: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame([{"nivel": "partido", "partido_id": p, "P_p": pp, "mediana_d1": m,
                          "iqr_d1": i} for p, pp, m, i in filas])


def test_posicion_vs_d1_clasifica_divergencias():
    pos = posicion([("p1", 0.0, 0.02, 0.0), ("p2", 0.5, -0.1, 0.9), ("p3", 0.5, -0.1, 0.0),
                    ("p4", 0.5, -0.1, 0.0), ("p5", None, -0.1, 0.0)])
    res = pd.DataFrame({"partido_id": ["p1", "p2", "p3", "p4"],
                        "mediana_ai": [1.0, 1.0, 0.6, 1.0]})
    clases = {r["partido_id"]: r["clase"]
              for r in conciliador(posicion=pos, resumen_cohesion=res).posicion_vs_d1()}
    assert clases == {"p1": "consistente",
                      "p2": "divergente_explicada_por_heterogeneidad",  # RIQ alto
                      "p3": "divergente_explicada_por_heterogeneidad",  # cohesión baja
                      "p4": "divergente_pendiente",
                      "p5": "sin_P_p"}


def test_salidas_faltantes_quedan_pendientes():
    faltan = conciliador(bcall_diputados=pd.DataFrame(), universo_por_metodo=None
                         ).salidas_faltantes()
    assert {c.control for c in faltan} == {"salida_bcall_diputados",
                                           "salida_universo_por_metodo"}
    assert {c.estado for c in faltan} == {"pendiente"}


def test_afinidad_detecta_covotos_incoherentes():
    mascara = pd.DataFrame({"diputado_id": ["A", "B", "C"], "j1": [True, True, True],
                            "j2": [True, True, False]})
    afin = pd.DataFrame({"diputado_i": ["A", "A", "B"], "diputado_j": ["B", "C", "C"],
                         "n_covotos": [2, 1, 2], "acuerdo": [1.0, 1.0, 0.5],
                         "hamming": [0.0, 0.0, 0.5], "incluido": ["True", "False", "True"]})
    estados = {c.control: (c.estado, c.n_comparado)
               for c in conciliador(mascara=mascara, afinidad=afin).controles_afinidad()}
    assert estados["pares_afinidad"] == ("coincide", 3)
    assert estados["covotos_vs_mascara"] == ("pendiente", 1)  # B-C tiene 1 co-voto, no 2
    assert estados["acuerdo_mas_hamming"][0] == "coincide"


def test_sensibilidad_debe_partir_de_los_resultados_publicados():
    part = pd.DataFrame({"diputado_id": list("ABCDE"),
                         "supera_umbral_bcall": ["True", "True", "False", "True", "True"]})
    pos = posicion([("p1", -0.5, -0.5, 0.0), ("p2", 0.9, 0.9, 0.0)])
    sens = pd.DataFrame({
        "metodo": ["perfil_partidario_P_p"] * 2 + ["bcall_d1"] * 4,
        "id": ["p1", "p2", "A", "B", "D", "E"],
        "valor_base": [-0.5, 0.8, -1.0, -0.7, 0.9, 0.9]})
    estados = {c.control: (c.estado, c.n_comparado) for c in conciliador(
        participacion=part, posicion=pos, sensibilidad=sens).controles_sensibilidad()}
    assert estados["sensibilidad_base_P_p"] == ("pendiente", 1)  # p2: 0,9 frente a 0,8
    assert estados["sensibilidad_diputados_d1"] == ("coincide", 4)  # C bajo el filtro


def test_partidos_cambiantes():
    afil = pd.DataFrame({"diputado_id": ["A", "A", "B"], "partido_alias": ["p1", "p2", "p1"]})
    assert conciliador(afiliacion=afil).partidos_cambiantes() == [
        {"diputado_id": "A", "partidos": ["p1", "p2"]}]


def test_corte_real_solo_quedan_pendientes_las_salidas_faltantes():
    conciliador_real, _ = desde_repositorio(RAIZ)
    r = conciliador_real.conciliar()
    pendientes = {c["control"] for c in r["controles"] if c["estado"] == "pendiente"}
    assert pendientes <= {"salida_bcall_diputados", "salida_universo_por_metodo"}
    assert r["descomposicion_universo_bcall"]["n_universo_esperado"] == 1728
    assert r["partidos_cambiantes"]["n_diputados"] == 20
    assert all(c["causa"] and c["decision"] for c in r["controles"]
               if c["estado"] == "explicada")
