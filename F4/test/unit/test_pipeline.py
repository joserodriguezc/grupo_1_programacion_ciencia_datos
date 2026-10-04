"""Pipeline de F4 (D01): orden de los pasos, validación de entradas y del contrato."""

import pytest

from F4.src.analisis import pipeline


def test_ejecuta_los_pasos_pedidos_en_el_orden_de_dependencia(monkeypatch, tmp_path) -> None:
    llamados = []
    monkeypatch.setattr(pipeline, "PASOS", {nombre: (lambda raiz, n=nombre: llamados.append(n))
                                            for nombre in pipeline.PASOS})
    ejecutados = pipeline.ejecutar(tmp_path, pasos=["universo", "matrices", "auditoria"],
                                   validar=False)
    assert ejecutados == llamados == ["auditoria", "matrices", "universo"]


def test_paso_desconocido_es_error(tmp_path) -> None:
    with pytest.raises(pipeline.ErrorPipeline, match="Pasos desconocidos"):
        pipeline.ejecutar(tmp_path, pasos=["no_existe"])


def test_exige_las_salidas_de_bcall_de_F4_02(tmp_path) -> None:
    with pytest.raises(pipeline.ErrorPipeline, match="F4_02"):
        pipeline.PASOS["universo"](tmp_path)


def test_valida_el_contrato_al_terminar(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline, "PASOS", {"auditoria": lambda raiz: None})
    with pytest.raises(pipeline.ErrorPipeline, match="fuera del contrato"):
        pipeline.ejecutar(tmp_path)


def test_la_exportacion_es_el_ultimo_paso() -> None:
    assert list(pipeline.PASOS)[-2:] == ["conciliacion", "exportacion"]
