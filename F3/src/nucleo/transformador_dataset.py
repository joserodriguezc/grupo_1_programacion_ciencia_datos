"""Transforma las cinco entradas interim en las tablas procesadas de F3.

La tabla observada conserva la fuente. La tabla analítica replica las decisiones
documentadas en F2_03, con selectores históricos que verifican coincidencia única.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .configuracion_validacion import construir_validador_interim
from .normalizacion import normalizar_fecha, normalizar_identificador, normalizar_total
from .validador_integracion import ValidadorIntegracion


@dataclass(frozen=True)
class ResultadoTransformacion:
    tablas: Mapping[str, pd.DataFrame]
    reporte_calidad: pd.DataFrame


def _seleccionar(
    tabla: pd.DataFrame, diputado: str, partido: str, inicio: str, termino: str
) -> object:
    """Selecciona exactamente un registro histórico, independientemente del orden."""
    mascara = (
        tabla["diputado_id"].eq(diputado)
        & tabla["partido_id"].eq(partido)
        & tabla["fecha_inicio"].eq(pd.Timestamp(inicio))
        & tabla["fecha_termino"].eq(pd.Timestamp(termino))
    )
    indices = tabla.index[mascara].tolist()
    if len(indices) != 1:
        raise ValueError(
            f"Se esperaba una militancia única para {diputado}/{partido} "
            f"{inicio}–{termino}; encontradas {len(indices)}"
        )
    return indices[0]


class TransformadorDataset:
    """Valida las entradas y reproduce las cinco tablas de datos de F2_03."""

    def transformar(self, tablas: Mapping[str, pd.DataFrame]) -> ResultadoTransformacion:
        nombres = {"periodos", "diputados", "militancias", "proyecto_ley", "detalle_votaciones"}
        if set(tablas) != nombres:
            raise ValueError(f"Entradas requeridas: {sorted(nombres)}; recibidas: {sorted(tablas)}")

        validador = construir_validador_interim()
        integracion = ValidadorIntegracion()
        reportes = [validador.validar(nombre, tablas[nombre]) for nombre in sorted(nombres)]
        reportes.extend(
            (
                integracion.validar_referencias_interim(tablas),
                integracion.validar_consistencia_interim(tablas),
                integracion.validar_periodos_y_totales_interim(tablas),
            )
        )
        errores = [f"{r.tabla}: {e.regla}: {e.detalle}" for r in reportes for e in r.errores]
        if errores:
            raise ValueError("Validación interim fallida:\n" + "\n".join(errores))

        calidad: list[dict[str, object]] = []

        def observar(tabla: str, diputado: str | None, tipo: str, detalle: str, regla: str) -> None:
            calidad.append(
                {
                    "tabla": tabla,
                    "diputado_id": diputado,
                    "tipo": tipo,
                    "descripcion": detalle,
                    "regla_aplicada": regla,
                }
            )

        diputados = tablas["diputados"].copy(deep=True)
        militancias = tablas["militancias"].copy(deep=True)
        detalle = tablas["detalle_votaciones"].copy(deep=True)
        proyecto = tablas["proyecto_ley"].copy(deep=True)
        periodos = tablas["periodos"].copy(deep=True)

        for df, columnas in (
            (diputados, ("diputado_id", "periodo_id")),
            (militancias, ("diputado_id", "partido_id")),
            (detalle, ("diputado_id", "votacion_id")),
            (proyecto, ("numero_boletin", "Id")),
            (periodos, ("periodo_id",)),
        ):
            for col in columnas:
                df[col] = normalizar_identificador(df[col], col)

        for df, obligatorias, opcionales in (
            (diputados, ("fecha_nacimiento", "fecha_inicio_periodo"), ("fecha_termino_periodo",)),
            (militancias, ("fecha_inicio",), ("fecha_termino",)),
            (detalle, ("fecha",), ()),
            (proyecto, ("Fecha",), ()),
            (periodos, ("fecha_inicio", "fecha_termino"), ()),
        ):
            for col in obligatorias:
                df[col] = normalizar_fecha(df[col], col)
            for col in opcionales:
                df[col] = normalizar_fecha(df[col], col, permitir_nulos=True)

        for df, columnas in (
            (detalle, ("total_si", "total_no", "total_abstencion", "total_dispensado")),
            (proyecto, ("TotalSi", "TotalNo", "TotalAbstencion", "TotalDispensado")),
        ):
            for col in columnas:
                df[col] = normalizar_total(df[col], col)

        # La anomalía del período de diputados se documenta sin reinterpretarla.
        fecha_frecuente = diputados["fecha_inicio_periodo"].mode().iloc[0]
        for _, fila in diputados.loc[
            diputados["fecha_inicio_periodo"].eq(fecha_frecuente)
        ].iterrows():
            observar(
                "diputados",
                fila["diputado_id"],
                "anomalia_temporal",
                "fecha_inicio_periodo no coincide con la semántica esperada; "
                "no se usa como vigencia.",
                "Conservar el dato original; usar periodos.csv para vigencias.",
            )

        ids_con_voto = set(detalle["diputado_id"])
        diputados["tiene_votos"] = diputados["diputado_id"].isin(ids_con_voto)
        for _, fila in diputados.loc[~diputados["tiene_votos"]].iterrows():
            observar(
                "diputados",
                fila["diputado_id"],
                "sin_votos",
                "No registra votos en las votaciones analizadas.",
                "Conservar en la dimensión; no crear votos artificiales.",
            )
        diputados_procesados = diputados.drop(columns=["rut", "rut_dv", "nombre2"])

        observadas = militancias.copy(deep=True)
        invalido = observadas["fecha_termino"].notna() & (
            observadas["fecha_termino"] < observadas["fecha_inicio"]
        )
        observadas["intervalo_valido"] = ~invalido
        observadas["observacion_calidad"] = None
        for _, fila in observadas.loc[invalido].iterrows():
            observar(
                "militancias",
                fila["diputado_id"],
                "intervalo_invertido",
                f"{fila['partido_id']}: término anterior al inicio.",
                "Conservar en observadas; excluir de analíticas.",
            )

        analiticas = observadas.loc[~invalido].copy()
        analiticas["fecha_termino_original"] = analiticas["fecha_termino"]
        analiticas["regla_asignacion_partido"] = "vigencia_temporal_estandar"
        fin_periodo = periodos.loc[periodos["periodo_id"].eq("10"), "fecha_termino"]
        if len(fin_periodo) != 1:
            raise ValueError("Debe haber exactamente un período legislativo 10")

        excluir = [
            ("1017", "UDI", "2018-03-11 00:00:00", "2025-03-18 23:59:59"),
            ("1017", "IND", "2020-09-29 00:00:00", "2022-03-10 23:59:59"),
            ("1017", "IND", "2025-03-19 00:00:00", "2026-03-10 23:59:59"),
            ("1114", "IND", "2022-06-13 00:00:00", "2024-07-02 23:59:59"),
            ("1114", "FA", "2024-07-03 00:00:00", "2026-03-10 23:59:59"),
            ("1114", "FA", "2026-03-11 00:00:00", "2030-03-10 23:59:59"),
            ("1180", "IND", "2024-05-31 00:00:00", "2026-03-10 23:59:59"),
        ]
        for diputado, partido, inicio, termino in excluir:
            indice = _seleccionar(analiticas, diputado, partido, inicio, termino)
            observar(
                "militancias",
                diputado,
                "excepcion_particular",
                f"Excluir {partido} ({inicio}–{termino}) de la línea analítica.",
                "Decisión analítica documentada en F2_03.",
            )
            analiticas = analiticas.drop(index=indice)

        for diputado, partido, inicio, termino, regla in (
            (
                "1114",
                "FRVS",
                "2022-03-11 00:00:00",
                "2023-06-12 23:59:59",
                "extension_caso_particular_bugueno",
            ),
            (
                "1180",
                "RD",
                "2022-03-11 00:00:00",
                "2024-05-30 23:59:59",
                "extension_caso_particular_veloso",
            ),
        ):
            indice = _seleccionar(analiticas, diputado, partido, inicio, termino)
            analiticas.loc[indice, "fecha_termino"] = fin_periodo.iloc[0]
            analiticas.loc[indice, "regla_asignacion_partido"] = regla

        liberal = analiticas["partido_id"].eq("LIBERAL")
        for _, fila in analiticas.loc[liberal].iterrows():
            observar(
                "militancias",
                fila["diputado_id"],
                "homologacion_catalogo",
                "LIBERAL homologado a PL.",
                "Solo en la línea analítica; conservar observadas.",
            )
        analiticas.loc[liberal, ["partido_id", "partido_alias"]] = "PL"

        # La tabla analítica debe tener una línea temporal sin ambigüedad.
        from F2.src.validaciones import detectar_solapamientos_vigencia

        conflictos = detectar_solapamientos_vigencia(analiticas)
        if not conflictos.empty:
            raise ValueError(f"Quedan {len(conflictos)} vigencias analíticas superpuestas")

        detalle_procesado = detalle.drop(columns=["nombre2"])
        resultado = {
            "diputados_procesados": diputados_procesados,
            "militancias_observadas": observadas,
            "militancias_analiticas": analiticas,
            "detalle_votaciones_procesado": detalle_procesado,
            "proyecto_ley_procesado": proyecto,
        }
        return ResultadoTransformacion(
            resultado,
            pd.DataFrame(
                calidad, columns=["tabla", "diputado_id", "tipo", "descripcion", "regla_aplicada"]
            ),
        )
