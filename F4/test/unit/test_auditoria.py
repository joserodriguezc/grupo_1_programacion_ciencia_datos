"""A02: cada regla detecta su error con datos mínimos; el corte limpio aprueba G0."""

import json

import pandas as pd
import pytest

from F4.src.analisis.auditoria import AuditorEntrada, Hallazgo, guardar_json

FECHA = "2023-05-08 19:05:22"
TEXTO = {"1": "Afirmativo", "0": "En Contra", "2": "Abstención"}


def fila(dip, vot, cod, partido="P1", **extra):
    base = {
        "diputado_id": dip, "votacion_id": vot, "fecha": FECHA,
        "opcion_codigo": cod, "opcion_voto": TEXTO.get(cod, "Otro"),
        "total_si": "1", "total_no": "1", "total_abstencion": "1", "total_dispensado": "0",
        "quorum_codigo": "1", "resultado_codigo": "1", "tipo_votacion_proyecto_ley": "Particular",
        "numero_boletin": "11092-07", "periodo_id": "10", "partido_id": partido,
        "partido_nombre": partido, "partido_alias": partido, "nombre": f"N{dip}",
        "apellido_paterno": "A", "apellido_materno": "B", "fecha_nacimiento": "1970-01-01",
        "sexo_valor": "1", "articulo": "Art. 1",
    }
    return base | extra


@pytest.fixture
def limpio():
    t = pd.DataFrame([fila("1", "10", "1"), fila("2", "10", "0"), fila("3", "10", "2")],
                     dtype="string")
    detalle = t[["diputado_id", "votacion_id", "opcion_codigo"]].copy()
    proyecto = pd.DataFrame({"Id": ["10"], "TotalSi": ["1"], "TotalNo": ["1"],
                             "TotalAbstencion": ["1"]}, dtype="string")
    mil = pd.DataFrame({"diputado_id": ["1", "2", "3"], "partido_id": ["P1"] * 3,
                        "fecha_inicio": ["2022-03-11 00:00:00"] * 3,
                        "fecha_termino": ["2026-03-10 23:59:59"] * 3}, dtype="string")
    padron = pd.DataFrame({"diputado_id": ["1", "2", "3"]}, dtype="string")
    manifiesto = {
        "entrada": {"sha256": "abc", "filas": 3, "columnas": t.shape[1], "votaciones": 1},
        "validacion": {"estado": "ok"},
        "reproducibilidad": {"commit_git": "022d4d0"},
    }
    return dict(tabla=t, detalle=detalle, proyecto=proyecto, militancias=mil, padron=padron,
                manifiesto=manifiesto, sha256_tabla="abc")


def auditar(datos, **cambios):
    datos = datos | cambios
    tabla = datos.pop("tabla")
    esperado = {"filas": len(tabla), "columnas": tabla.shape[1], "votaciones": 1}
    return AuditorEntrada(tabla, esperado=esperado, **datos).auditar()


def estados(reporte):
    return {h.regla: h.estado for h in reporte.hallazgos}


def test_corte_limpio_aprueba_g0(limpio):
    r = auditar(limpio)
    assert r.aprobado, [h for h in r.hallazgos if h.estado != "ok"]
    assert set(estados(r).values()) == {"ok"}


def test_sin_manifiesto_bloquea(limpio):
    r = auditar(limpio, manifiesto=None)
    assert estados(r)["R00_manifiesto_corte"] == "error" and not r.aprobado


def test_hash_distinto_del_manifiesto_bloquea(limpio):
    assert estados(auditar(limpio, sha256_tabla="otro"))["R00_manifiesto_corte"] == "error"


def test_dimensiones_o_validacion_distintas_del_manifiesto_bloquean(limpio):
    otro = limpio["manifiesto"] | {"entrada": limpio["manifiesto"]["entrada"] | {"filas": 4}}
    assert estados(auditar(limpio, manifiesto=otro))["R00_manifiesto_corte"] == "error"
    otro = limpio["manifiesto"] | {"validacion": {"estado": "variacion_justificada"}}
    assert estados(auditar(limpio, manifiesto=otro))["R00_manifiesto_corte"] == "error"


def test_duplicado_diputado_votacion(limpio):
    t = pd.concat([limpio["tabla"], limpio["tabla"].iloc[[0]]], ignore_index=True)
    r = auditar(limpio, tabla=t)
    h = next(h for h in r.hallazgos if h.regla == "R03_clave_unica")
    assert h.estado == "error" and h.n_afectados == 2


def test_codigo_desconocido_o_texto_inconsistente(limpio):
    t = limpio["tabla"].copy()
    t.loc[0, "opcion_codigo"] = "3"
    assert estados(auditar(limpio, tabla=t))["R05_dominio_voto"] == "error"
    t = limpio["tabla"].copy()
    t.loc[2, "opcion_voto"] = "Afirmativo"  # código 2 con texto de Sí
    assert estados(auditar(limpio, tabla=t))["R05_dominio_voto"] == "error"


def test_totales_no_cuadran(limpio):
    t = limpio["tabla"].copy()
    t["total_si"] = "2"
    e = estados(auditar(limpio, tabla=t))
    assert e["R07_totales"] == "error" and e["R10_contraste_proyecto"] == "error"


def test_atributo_de_votacion_variable(limpio):
    t = limpio["tabla"].copy()
    t.loc[1, "fecha"] = "2023-05-09 10:00:00"
    assert estados(auditar(limpio, tabla=t))["R06_atributos_votacion"] == "error"


def test_voto_distinto_del_detalle_f3(limpio):
    detalle = limpio["detalle"].copy()
    detalle.loc[0, "opcion_codigo"] = "0"
    assert estados(auditar(limpio, detalle=detalle))["R09_contraste_detalle"] == "error"


def test_partido_distinto_del_vigente_a_la_fecha(limpio):
    t = limpio["tabla"].copy()
    t.loc[0, "partido_id"] = "P2"
    assert estados(auditar(limpio, tabla=t))["R11_militancia_fecha"] == "error"


def test_dos_militancias_vigentes_es_ambiguo(limpio):
    extra = pd.DataFrame({"diputado_id": ["1"], "partido_id": ["P9"],
                          "fecha_inicio": ["2023-01-01 00:00:00"],
                          "fecha_termino": ["2023-12-31 00:00:00"]}, dtype="string")
    mil = pd.concat([limpio["militancias"], extra], ignore_index=True)
    assert estados(auditar(limpio, militancias=mil))["R11_militancia_fecha"] == "error"


def test_diputado_fuera_del_padron_y_padron_sin_votos(limpio):
    padron = pd.DataFrame({"diputado_id": ["1", "2", "99"]}, dtype="string")
    e = estados(auditar(limpio, padron=padron))
    assert e["R13_padron"] == "error" and e["R13_padron_sin_votos"] == "advertencia"


def test_nulo_explicado_solo_si_el_patron_se_verifica(limpio):
    t = limpio["tabla"].copy()
    t["tipo_votacion_proyecto_ley"] = "General"
    t["articulo"] = pd.NA
    assert estados(auditar(limpio, tabla=t))["R14_nulos_articulo"] == "advertencia"
    t["tipo_votacion_proyecto_ley"] = "Particular"
    assert estados(auditar(limpio, tabla=t))["R14_nulos_articulo"] == "error"


def test_falta_columna_obligatoria(limpio):
    r = auditar(limpio, tabla=limpio["tabla"].drop(columns="partido_id"))
    assert [h.regla for h in r.hallazgos] == ["R02_columnas"] and not r.aprobado


def test_advertencia_exige_explicacion():
    with pytest.raises(ValueError):
        Hallazgo("R99", "advertencia", "sin explicar")


def test_no_modifica_la_tabla_de_entrada(limpio):
    copia = limpio["tabla"].copy()
    auditar(limpio)
    pd.testing.assert_frame_equal(limpio["tabla"], copia)


def test_json_determinista(limpio, tmp_path):
    r = auditar(limpio)
    a = guardar_json(r, tmp_path / "a.json").read_bytes()
    b = guardar_json(auditar(limpio), tmp_path / "b.json").read_bytes()
    assert a == b
    assert json.loads(a)["puerta_g0"] == "aprobada"


def test_perfil_desempata_partidos_por_nombre(limpio):
    t = pd.DataFrame([fila("1", "10", "1", "ZZ"), fila("2", "10", "0", "AA"),
                      fila("3", "10", "2", "MM")], dtype="string")
    r = auditar(limpio, tabla=t)
    assert [p["partido_alias"] for p in r.perfil["partidos_detalle"]] == ["AA", "MM", "ZZ"]
