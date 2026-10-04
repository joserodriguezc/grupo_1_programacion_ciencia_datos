"""analisis.toml y decisiones_metodologicas.json deben describir el mismo protocolo."""

import hashlib
import json
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
TOML = RAIZ / "F4/config/analisis.toml"
REGISTRO = json.loads(
    (RAIZ / "F4/data/reports/decisiones_metodologicas.json").read_text(encoding="utf-8"))
CONFIG = tomllib.loads(TOML.read_text(encoding="utf-8"))


def test_registro_copia_los_parametros_y_el_hash_del_toml() -> None:
    asociada = REGISTRO["configuracion_asociada"]
    # El hash se calcula con saltos LF: no depende del sistema operativo del checkout.
    contenido = TOML.read_bytes().replace(b"\r\n", b"\n")

    assert asociada["sha256"] == hashlib.sha256(contenido).hexdigest()
    assert asociada["parametros"] == CONFIG


def test_versiones_coinciden_con_el_historial() -> None:
    version = CONFIG["protocolo"]["version"]
    assert REGISTRO["version_registro"] == REGISTRO["configuracion_asociada"]["version"] == version
    assert REGISTRO["historial"][-1]["version"] == version


def test_parametros_citados_por_las_decisiones_existen_en_el_toml() -> None:
    for decision in REGISTRO["decisiones"]:
        for clave, valor in decision.get("parametros_toml", {}).items():
            seccion, _, nombre = clave.partition(".")
            assert CONFIG[seccion][nombre] == valor, f"{decision['id']}: {clave}"
