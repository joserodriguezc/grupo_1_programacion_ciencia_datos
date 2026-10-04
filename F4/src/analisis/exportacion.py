"""Exportación de la entrega de F4: manifiesto_entrega.json.

Registra qué se entrega y en qué estado: versión y estado del protocolo, puertas de aprobación,
clasificación de los resultados, corte F3 de origen y, para cada entregable (tablas y
reportes del contrato de datos, figuras, notebooks y configuración), su ruta, tipo,
productor, tamaño y SHA-256.

El manifiesto es determinista: no incluye fecha ni commit propio, de modo que solo cambia
si cambia algún entregable. Exige que el protocolo esté aprobado cuando
[control].exigir_protocolo_aprobado_para_publicar es verdadero, y que todas las salidas
cumplan el contrato. Es el último paso del pipeline.

Uso: python -m F4.src.analisis.exportacion
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from F4.src.analisis import contratos
from F4.src.analisis.corte import calcular_sha256

RAIZ_REPOSITORIO = contratos.RAIZ_REPOSITORIO
SALIDA = contratos.SALIDAS["manifiesto_entrega"].ruta
# Carpeta de figuras → notebook que las genera.
FIGURAS_POR_NOTEBOOK = {
    "datos_cobertura": "F4_01_datos_y_cobertura.ipynb", "bcall": "F4_02_bcall.ipynb",
    "partidos_afinidad": "F4_03_metricas_partidos_afinidad.ipynb",
    "comparacion": "F4_04_clustering_pca.ipynb", "cohesion": "F4_05_cohesion.ipynb",
    "comunicacion": "F4_06_comunicacion.ipynb",
}
CONFIGURACION = ("F4/config/analisis.toml", "F4/config/diccionario_votos.toml",
                 "F4/docs/clasificacion_ideologica_partidos_chilenos.json")


class ErrorExportacion(RuntimeError):
    """La entrega no puede exportarse con el protocolo o las salidas actuales."""


def _entregable(raiz: Path, ruta: str, tipo: str, productor: str) -> dict:
    archivo = raiz / ruta
    return {"ruta": ruta, "tipo": tipo, "productor": productor,
            "bytes": archivo.stat().st_size, "sha256": calcular_sha256(archivo)}


def entregables(raiz: Path | str = RAIZ_REPOSITORIO) -> list[dict]:
    """Lista ordenada de entregables con su hash; no incluye el propio manifiesto."""
    raiz = Path(raiz)
    lista = [_entregable(raiz, salida.ruta, "tabla" if salida.es_tabla else "reporte",
                         salida.productor)
             for nombre, salida in contratos.SALIDAS.items() if nombre != "manifiesto_entrega"]
    lista += [_entregable(raiz, p.relative_to(raiz).as_posix(), "figura",
                          FIGURAS_POR_NOTEBOOK.get(p.parent.name, "sin productor registrado"))
              for p in sorted((raiz / "F4/figures").rglob("*.png"))]
    lista += [_entregable(raiz, p.relative_to(raiz).as_posix(), "notebook", p.name)
              for p in sorted((raiz / "F4/notebooks").glob("F4_0*.ipynb"))]
    lista += [_entregable(raiz, ruta, "configuracion", "protocolo metodológico")
              for ruta in CONFIGURACION if (raiz / ruta).is_file()]
    return sorted(lista, key=lambda e: (e["tipo"], e["ruta"]))


def construir_manifiesto(raiz: Path | str = RAIZ_REPOSITORIO) -> dict:
    raiz = Path(raiz)
    with (raiz / "F4/config/analisis.toml").open("rb") as archivo:
        config = tomllib.load(archivo)
    protocolo = config["protocolo"]
    if (config["control"]["exigir_protocolo_aprobado_para_publicar"]
            and protocolo["estado"] != "aprobado"):
        raise ErrorExportacion(f"El protocolo está en estado «{protocolo['estado']}»; "
                               "la entrega exige protocolo aprobado (PROTOCOLO_NO_APROBADO).")
    problemas = contratos.validar_repositorio(raiz, omitir=("manifiesto_entrega",))
    if problemas:
        raise ErrorExportacion("Salidas fuera del contrato:\n" + "\n".join(problemas))

    decisiones = json.loads((raiz / contratos.SALIDAS["decisiones"].ruta)
                            .read_text(encoding="utf-8"))
    corte = json.loads((raiz / contratos.SALIDAS["manifiesto_corte"].ruta)
                       .read_text(encoding="utf-8"))
    puertas = decisiones["puertas"]
    lista = entregables(raiz)
    return {
        "entrega": "F4",
        "generado_por": "F4/src/analisis/exportacion.py",
        "protocolo": {"version": protocolo["version"], "estado": protocolo["estado"]},
        "puertas": {nombre: puerta["estado"] for nombre, puerta in sorted(puertas.items())},
        "clasificacion_resultados": (puertas.get("clasificacion_resultados", {})
                                     .get("clasificacion_general")),
        "corte_f3": {"ruta": corte["entrada"]["ruta"], "sha256": corte["entrada"]["sha256"],
                     "commit_git": corte["reproducibilidad"]["commit_git"]},
        "resumen": {tipo: sum(e["tipo"] == tipo for e in lista)
                    for tipo in sorted({e["tipo"] for e in lista})},
        "entregables": lista,
    }


def main(raiz: Path | str = RAIZ_REPOSITORIO) -> dict:
    manifiesto = construir_manifiesto(raiz)
    destino = Path(raiz) / SALIDA
    destino.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
    print(f"Manifiesto de entrega: {len(manifiesto['entregables'])} entregables "
          f"{manifiesto['resumen']} → {SALIDA}")
    return manifiesto


if __name__ == "__main__":
    try:
        main()
    except ErrorExportacion as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
