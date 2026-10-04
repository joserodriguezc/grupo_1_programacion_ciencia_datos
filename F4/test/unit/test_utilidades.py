"""Criterio de comparación de salidas regeneradas (F4/test/utilidades.py)."""

from pathlib import Path

from F4.test.utilidades import diferencia_salida


def _csv(tmp_path: Path, nombre: str, texto: str) -> Path:
    ruta = tmp_path / nombre
    ruta.write_bytes(texto.encode("utf-8"))
    return ruta


def test_acepta_solo_el_ruido_de_punto_flotante(tmp_path: Path) -> None:
    base = _csv(tmp_path, "a.csv", "id,PC1\n803,1.0515337236696587\n815,0.9999999999999999\n")
    ruido = _csv(tmp_path, "b.csv", "id,PC1\n803,1.0515337236696585\n815,1.0\n")
    cambio = _csv(tmp_path, "c.csv", "id,PC1\n803,1.0515\n815,1.0\n")

    assert diferencia_salida(base, base) is None
    assert diferencia_salida(ruido, base) is None
    assert diferencia_salida(cambio, base) is not None


def test_textos_enteros_y_no_csv_son_exactos(tmp_path: Path) -> None:
    base = _csv(tmp_path, "a.csv", "id,estado\n803,OK\n")
    assert diferencia_salida(_csv(tmp_path, "b.csv", "id,estado\n804,OK\n"), base) is not None
    assert diferencia_salida(_csv(tmp_path, "c.csv", "id,estado\n803,NA_\n"), base) is not None

    json_base = _csv(tmp_path, "a.json", '{"x": 1.0}\n')
    assert diferencia_salida(_csv(tmp_path, "b.json", '{"x": 1.0000000000000002}\n'),
                             json_base) is not None
