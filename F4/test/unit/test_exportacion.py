"""Exportación de la entrega: manifiesto determinista, hashes y bloqueos."""

import shutil
from pathlib import Path

import pytest

from F4.src.analisis import exportacion
from F4.src.analisis.corte import calcular_sha256

RAIZ = Path(__file__).resolve().parents[3]


def test_manifiesto_del_repositorio() -> None:
    manifiesto = exportacion.construir_manifiesto(RAIZ)

    assert manifiesto["protocolo"]["estado"] == "aprobado"
    assert manifiesto["puertas"]["clasificacion_resultados"] == "aprobada"
    assert manifiesto["clasificacion_resultados"] == "descriptivo"
    tipos = manifiesto["resumen"]
    assert tipos["notebook"] == 6 and tipos["figura"] > 0 and tipos["tabla"] > 0
    rutas = [e["ruta"] for e in manifiesto["entregables"]]
    assert exportacion.SALIDA not in rutas
    assert len(rutas) == len(set(rutas))
    # Determinista: dos construcciones seguidas son idénticas.
    assert exportacion.construir_manifiesto(RAIZ) == manifiesto


def test_hash_y_tamano_de_cada_entregable() -> None:
    for entregable in exportacion.entregables(RAIZ)[:10]:
        archivo = RAIZ / entregable["ruta"]
        assert entregable["sha256"] == calcular_sha256(archivo)
        assert entregable["bytes"] == archivo.stat().st_size


def test_figuras_tienen_notebook_productor() -> None:
    figuras = [e for e in exportacion.entregables(RAIZ) if e["tipo"] == "figura"]
    assert {e["productor"] for e in figuras} <= set(exportacion.FIGURAS_POR_NOTEBOOK.values())


def _repo_minimo(tmp_path: Path, estado: str) -> Path:
    (tmp_path / "F4/config").mkdir(parents=True)
    texto = (RAIZ / "F4/config/analisis.toml").read_text(encoding="utf-8")
    texto = texto.replace('estado = "aprobado"', f'estado = "{estado}"', 1)
    (tmp_path / "F4/config/analisis.toml").write_text(texto, encoding="utf-8")
    return tmp_path


def test_bloquea_si_el_protocolo_no_esta_aprobado(tmp_path) -> None:
    raiz = _repo_minimo(tmp_path, "propuesta_pendiente_revision")
    with pytest.raises(exportacion.ErrorExportacion, match="PROTOCOLO_NO_APROBADO"):
        exportacion.construir_manifiesto(raiz)


def test_bloquea_si_faltan_salidas(tmp_path) -> None:
    raiz = _repo_minimo(tmp_path, "aprobado")
    shutil.copytree(RAIZ / "F4/data/processed", raiz / "F4/data/processed")
    with pytest.raises(exportacion.ErrorExportacion, match="fuera del contrato"):
        exportacion.construir_manifiesto(raiz)
