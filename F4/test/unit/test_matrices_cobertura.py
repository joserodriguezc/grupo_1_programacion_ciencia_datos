"""Pruebas de matrices, máscara, afiliación y participación observada."""

from pathlib import Path

import pandas as pd
import pytest

from F4.src.analisis.cobertura import ErrorCobertura, ParticipacionCorpus
from F4.src.analisis.matrices import (
    ConstructorMatrices,
    ErrorMatrices,
    validar_contra_auditoria,
)

RAIZ = Path(__file__).resolve().parents[3]


def _tabla_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diputado_id": [10, 10, 11, 11],
            "votacion_id": [100, 101, 100, 101],
            "fecha": [
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-01 10:00:00",
                "2026-01-02 10:00:00",
            ],
            "partido_id": [1, 2, 3, pd.NA],
            "partido_nombre": ["Partido A", "Partido B", "Independientes", pd.NA],
            "partido_alias": ["PA", "PB", "IND", pd.NA],
            "voto_binario": [1, pd.NA, 0, 1],
            "voto_ternario": [1, 0, -1, 1],
            "voto_nominal": ["Sí", "Abstención", "No", "Sí"],
            "observado": [True, True, True, True],
        }
    )


def test_construye_tres_matrices_y_mascara() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    assert resultado.binaria.shape == (2, 3)
    assert resultado.ternaria.shape == (2, 3)
    assert resultado.nominal.shape == (2, 3)
    assert resultado.mascara.shape == (2, 3)

    binaria = resultado.binaria.set_index("diputado_id")
    ternaria = resultado.ternaria.set_index("diputado_id")
    nominal = resultado.nominal.set_index("diputado_id")

    assert binaria.loc[10, 100] == 1
    assert pd.isna(binaria.loc[10, 101])
    assert ternaria.loc[10, 101] == 0
    assert nominal.loc[10, 101] == "Abstención"


def test_fila_faltante_no_se_interpreta_como_ausencia() -> None:
    tabla = _tabla_base().drop(index=3).reset_index(drop=True)
    resultado = ConstructorMatrices().construir(tabla)

    mascara = resultado.mascara.set_index("diputado_id")
    binaria = resultado.binaria.set_index("diputado_id")
    ternaria = resultado.ternaria.set_index("diputado_id")
    nominal = resultado.nominal.set_index("diputado_id")

    assert not bool(mascara.loc[11, 101])
    assert pd.isna(binaria.loc[11, 101])
    assert pd.isna(ternaria.loc[11, 101])
    assert pd.isna(nominal.loc[11, 101])


def test_abstencion_es_observada_y_ternaria_cero() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    mascara = resultado.mascara.set_index("diputado_id")
    binaria = resultado.binaria.set_index("diputado_id")
    ternaria = resultado.ternaria.set_index("diputado_id")

    assert bool(mascara.loc[10, 101])
    assert pd.isna(binaria.loc[10, 101])
    assert ternaria.loc[10, 101] == 0


def test_rechaza_clave_duplicada() -> None:
    tabla = pd.concat(
        [_tabla_base(), _tabla_base().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ErrorMatrices, match="debe ser única"):
        ConstructorMatrices().construir(tabla)


def test_afiliacion_extrae_solo_filas_existentes() -> None:
    tabla = _tabla_base().drop(index=3).reset_index(drop=True)
    resultado = ConstructorMatrices().construir(tabla)

    assert len(resultado.afiliacion) == len(tabla)

    clave = resultado.afiliacion[["diputado_id", "votacion_id"]]
    assert not clave.duplicated().any()

    # No se inventa la combinación 11 × 101 eliminada de la entrada.
    assert not (
        resultado.afiliacion["diputado_id"].eq(11)
        & resultado.afiliacion["votacion_id"].eq(101)
    ).any()


def test_afiliacion_conserva_cambio_historico() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    diputado = resultado.afiliacion[
        resultado.afiliacion["diputado_id"].eq(10)
    ].sort_values("votacion_id")

    assert diputado["partido_id"].tolist() == [1, 2]
    assert diputado["partido_nombre"].tolist() == ["Partido A", "Partido B"]


def test_afiliacion_conserva_nulos_e_independientes() -> None:
    resultado = ConstructorMatrices().construir(_tabla_base())

    independiente = resultado.afiliacion[
        (resultado.afiliacion["diputado_id"] == 11)
        & (resultado.afiliacion["votacion_id"] == 100)
    ].iloc[0]
    desconocida = resultado.afiliacion[
        (resultado.afiliacion["diputado_id"] == 11)
        & (resultado.afiliacion["votacion_id"] == 101)
    ].iloc[0]

    assert independiente["partido_nombre"] == "Independientes"
    assert pd.isna(desconocida["partido_id"])
    assert pd.isna(desconocida["partido_nombre"])
    assert pd.isna(desconocida["partido_alias"])


def test_corte_real_conserva_conteos_y_afiliaciones() -> None:
    votos = pd.read_csv(RAIZ / "F4/data/processed/votos_codificados.csv")
    resultado = ConstructorMatrices().construir(votos)

    assert resultado.n_diputados == 151
    assert resultado.n_votaciones == 15
    assert resultado.n_decisiones_observadas == 1993
    assert resultado.n_celdas_sin_registro == 272

    columnas = [c for c in resultado.mascara.columns if c != "diputado_id"]

    assert int(resultado.binaria[columnas].notna().sum().sum()) == 1925
    assert int(resultado.ternaria[columnas].notna().sum().sum()) == 1993
    assert int(resultado.nominal[columnas].notna().sum().sum()) == 1993

    assert len(resultado.afiliacion) == len(votos)
    assert not resultado.afiliacion.duplicated(
        ["diputado_id", "votacion_id"]
    ).any()


def test_concilia_con_auditoria_real() -> None:
    votos = pd.read_csv(RAIZ / "F4/data/processed/votos_codificados.csv")

    validar_contra_auditoria(
        votos,
        RAIZ / "F4/data/reports/auditoria_entrada.json",
    )


# ---------------------------------- Participación observada ----------------------------------

NOMINAL = {1: "Sí", 0: "Abstención", -1: "No"}
FECHAS = {"j1": "2023-05-08 10:00:00", "j2": "2023-05-08 11:00:00",
          "j3": "2023-05-08 12:00:00", "j4": "2024-08-26 10:00:00"}
# Ejemplo de referencia: C sin decisión en j4; B se abstiene en j4.
GUIA = {
    "A": ("p1", [-1, 1, -1, 1]), "B": ("p1", [-1, 1, -1, 0]), "C": ("p1", [-1, 1, 1, None]),
    "D": ("p2", [1, 1, 1, -1]), "E": ("p2", [1, 1, 1, -1]),
}


def tabla_guia(**cambios) -> pd.DataFrame:
    filas = []
    for dip, (partido, votos) in GUIA.items():
        for vot, v in zip(FECHAS, votos):
            if v is not None:
                filas.append({"diputado_id": dip, "votacion_id": vot, "fecha": FECHAS[vot],
                              "partido_id": partido, "partido_alias": partido,
                              "voto_nominal": NOMINAL[v], "observado": "True"})
    t = pd.DataFrame(filas, dtype="string")
    totales = t.groupby(["votacion_id", "voto_nominal"]).size().unstack(fill_value=0)
    for col, cat in (("total_si", "Sí"), ("total_no", "No"), ("total_abstencion", "Abstención")):
        serie = totales[cat] if cat in totales else 0
        t[col] = t["votacion_id"].map(serie).fillna(0).astype(int).astype(str)
    t["total_dispensado"] = "0"
    for col, valor in cambios.items():
        t[col] = valor
    return t


def calidad_guia() -> pd.DataFrame:
    """Forma de reporte_calidad.csv de F3: A con una excepción en militancias."""
    return pd.DataFrame([
        {"tabla": "militancias", "diputado_id": "A", "tipo": "excepcion_particular",
         "descripcion": "x", "regla_aplicada": "y"},
        {"tabla": "diputados", "diputado_id": "B", "tipo": "anomalia_temporal",
         "descripcion": "x", "regla_aplicada": "y"},
    ], dtype="string")


@pytest.fixture
def guia():
    return ParticipacionCorpus(tabla_guia(), calidad_guia(), umbral_bcall=0.10)


def test_participacion_es_votos_observados_sobre_votaciones_del_corpus(guia):
    p = guia.participacion_diputados().set_index("diputado_id")
    assert p.loc["C", "n_votos_observados"] == 3 and p.loc["C", "n_sin_registro"] == 1
    assert p.loc["C", "participacion_corpus"] == pytest.approx(0.75)
    assert p.loc["B", ["n_si", "n_no", "n_abst"]].tolist() == [1, 2, 1]
    assert (p["n_votaciones_corpus"] == 4).all()


def test_abstencion_cuenta_como_participacion(guia):
    p = guia.participacion_diputados().set_index("diputado_id")
    assert p.loc["B", "participacion_corpus"] == 1


def test_umbral_bcall_estrictamente_mayor():
    t = tabla_guia()
    t = t[~((t.diputado_id == "C") & t.votacion_id.isin(["j2", "j3"]))]  # C con 1 de 4
    p = ParticipacionCorpus(t, umbral_bcall=0.25).participacion_diputados()
    c = p.set_index("diputado_id").loc["C"]
    assert c["participacion_corpus"] == pytest.approx(0.25) and not c["supera_umbral_bcall"]


def test_celda_sin_registro_no_se_declara_ausencia(guia):
    faltan = guia.celdas_sin_registro()
    assert faltan[["diputado_id", "votacion_id", "estado"]].values.tolist() == [
        ["C", "j4", "sin_registro"]]


def test_conteos_partido_votacion_del_ejemplo(guia):
    c = guia.conteos_partido_votacion().set_index(["partido_id", "votacion_id"])
    assert c.loc[("p1", "j4"), ["Y", "N", "A", "T"]].tolist() == [1, 0, 1, 2]
    assert c.loc[("p1", "j3"), ["Y", "N", "A", "T"]].tolist() == [1, 2, 0, 3]
    assert c.loc[("p2", "j4"), ["Y", "N", "A", "T"]].tolist() == [0, 2, 0, 2]


def test_conciliacion_detecta_diferencias():
    assert set(ParticipacionCorpus(tabla_guia()).conciliacion()["estado"]) == {"coincide"}
    t = tabla_guia()
    t.loc[t.votacion_id == "j1", "total_si"] = "9"
    conc = ParticipacionCorpus(t).conciliacion().set_index("votacion_id")
    assert conc.loc["j1", "estado"] == "pendiente" and conc.loc["j1", "dif_total_si"] == 2 - 9
    assert set(conc.drop("j1")["estado"]) == {"coincide"}


def test_afiliacion_conserva_calidad_f3_sin_recalcular(guia):
    a = guia.afiliacion().set_index("diputado_id")
    assert set(a.loc["A", "calidad_militancia_f3"]) == {"excepcion_particular"}
    assert a.drop("A")["calidad_militancia_f3"].isna().all()  # solo tabla militancias
    assert set(a["estado_afiliacion"]) == {"resuelta"}
    p = guia.participacion_diputados().set_index("diputado_id")
    assert p.loc["A", "calidad_militancia_f3"] == "excepcion_particular"
    c = guia.conteos_partido_votacion().set_index(["partido_id", "votacion_id"])
    assert c.loc[("p1", "j1"), "n_con_observacion_f3"] == 1


def test_afiliacion_no_resuelta_queda_visible():
    t = tabla_guia()
    t.loc[(t.diputado_id == "D") & (t.votacion_id == "j1"), ["partido_id", "partido_alias"]] = (
        pd.NA)
    corpus = ParticipacionCorpus(t)
    c = corpus.conteos_partido_votacion().set_index(["partido_id", "votacion_id"])
    assert c.loc[("<sin_partido>", "j1"), ["T", "n_afiliacion_no_resuelta"]].tolist() == [1, 1]
    assert c.loc[("p2", "j1"), "T"] == 1
    assert corpus.reporte_conciliacion()["resumen"]["n_afiliacion_no_resuelta"] == 1


def test_rechaza_clave_duplicada_y_observado_incoherente():
    t = tabla_guia()
    with pytest.raises(ErrorCobertura):
        ParticipacionCorpus(pd.concat([t, t.iloc[[0]]], ignore_index=True))
    t.loc[0, "voto_nominal"] = pd.NA
    with pytest.raises(ErrorCobertura):
        ParticipacionCorpus(t)


def test_reporte_informa_filtro_sin_eliminar_ni_hablar_de_asistencia(guia):
    r = guia.reporte_conciliacion({"corte_sha256": "abc"})
    assert r["resumen"]["n_celdas_sin_registro"] == 1
    assert r["resumen"]["n_decisiones_observadas"] == 19
    assert r["conciliacion"]["votaciones_pendientes"] == 0
    assert r["trazabilidad"] == {"corte_sha256": "abc"}
    assert "No es tasa de asistencia" in r["advertencia"]
    assert len(guia.participacion_diputados()) == 5  # nadie se elimina


def test_exportar_cuatro_salidas_deterministas(guia, tmp_path):
    a = guia.exportar(tmp_path / "a")
    b = ParticipacionCorpus(tabla_guia(), calidad_guia()).exportar(tmp_path / "b")
    assert sorted(p.name for p in a.values()) == [
        "celdas_sin_registro.csv", "conciliacion_conteos.json",
        "conteos_partido_votacion.csv", "participacion_observada.csv"]
    assert all(a[k].read_bytes() == b[k].read_bytes() for k in a)


def test_celdas_sin_registro_coinciden_con_mascara_de_matrices() -> None:
    """Corte real: las celdas sin registro son las celdas falsas de la máscara."""
    votos = pd.read_csv(RAIZ / "F4/data/processed/votos_codificados.csv", dtype="string")
    faltan = ParticipacionCorpus(votos).celdas_sin_registro()
    mascara = ConstructorMatrices().construir(
        pd.read_csv(RAIZ / "F4/data/processed/votos_codificados.csv")).mascara
    largo = mascara.melt(id_vars="diputado_id", var_name="votacion_id", value_name="obs")
    falsas = {(str(d), str(v)) for d, v, o in largo.itertuples(index=False) if not o}
    assert len(faltan) == 272
    assert set(zip(faltan["diputado_id"], faltan["votacion_id"])) == falsas
