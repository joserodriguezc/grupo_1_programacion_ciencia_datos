"""Ejecuta y conserva las validaciones interim de F3."""

from pathlib import Path

import pandas as pd

from F3.src.nucleo.configuracion_validacion import construir_validador_interim
from F3.src.nucleo.persistencia_validacion import guardar_reporte_json
from F3.src.nucleo.validador_integracion import ValidadorIntegracion

INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"

RUTAS = {
    "periodos": INTERIM / "periodos.csv",
    "diputados": INTERIM / "diputados.csv",
    "militancias": INTERIM / "militancias.csv",
    "proyecto_ley": (
        INTERIM / "VotacionesPorProyectoDeLey" / "proyecto_ley.csv"
    ),
    "detalle_votaciones": (
        INTERIM / "votaciones" / "detalle_votaciones.csv"
    ),
}


def main() -> None:
    tablas = {}
    reportes = []
    validador = construir_validador_interim()

    for nombre, ruta in RUTAS.items():
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta el CSV {nombre}: {ruta}")

        tablas[nombre] = pd.read_csv(
            ruta, dtype="string", encoding="utf-8-sig"
        )
        reporte = validador.validar(nombre, tablas[nombre])
        destino = guardar_reporte_json(
            reporte,
            archivo_entrada=ruta,
        )
        reportes.append(reporte)
        print(f"{nombre}: {len(reporte.registros)} reglas → {destino}")

    integracion = ValidadorIntegracion()
    controles = (
        integracion.validar_referencias_interim(tablas),
        integracion.validar_consistencia_interim(tablas),
        integracion.validar_periodos_y_totales_interim(tablas),
    )

    for reporte in controles:
        destino = guardar_reporte_json(
            reporte,
            archivos_entrada=RUTAS,
        )
        reportes.append(reporte)
        print(f"{reporte.tabla}: {len(reporte.registros)} reglas → {destino}")

    rechazados = [r.tabla for r in reportes if not r.aprobado]
    if rechazados:
        raise SystemExit(
            f"Validación no aprobada: {', '.join(rechazados)}"
        )

    print("Todos los reportes aprobaron.")


if __name__ == "__main__":
    main()