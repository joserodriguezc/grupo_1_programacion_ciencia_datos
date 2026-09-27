"""Validaciones que necesitan relacionar más de una tabla."""

from collections.abc import Mapping

import pandas as pd

from .registro_validacion import RegistroValidacion, ReporteValidacion


class ValidadorIntegracion:
    """Comprueba relaciones sin modificar DataFrames ni crear la big table."""

    TABLAS_INTERIM = frozenset({
        "periodos",
        "diputados",
        "militancias",
        "proyecto_ley",
        "detalle_votaciones",
    })

    REFERENCIAS_INTERIM = (
        ("detalle_votaciones", "votacion_id", "proyecto_ley", "Id"),
        ("proyecto_ley", "Id", "detalle_votaciones", "votacion_id"),
        ("detalle_votaciones", "diputado_id", "diputados", "diputado_id"),
        ("militancias", "diputado_id", "diputados", "diputado_id"),
        ("diputados", "periodo_id", "periodos", "periodo_id"),
        ("detalle_votaciones", "diputado_id", "militancias", "diputado_id"),
    )

    CAMPOS_VOTACION = (
        "descripcion",
        "fecha",
        "total_si",
        "total_no",
        "total_abstencion",
        "total_dispensado",
        "quorum_codigo",
        "quorum",
        "resultado_codigo",
        "resultado",
        "tipo_codigo",
        "tipo",
    )

    EQUIVALENCIAS_PROYECTO = {
        "Descripcion": "descripcion",
        "Fecha": "fecha",
        "TotalSi": "total_si",
        "TotalNo": "total_no",
        "TotalAbstencion": "total_abstencion",
        "TotalDispensado": "total_dispensado",
        "Quorum": "quorum",
        "Resultado": "resultado",
        "Tipo": "tipo",
    }

    def validar_referencias_interim(
        self, tablas: Mapping[str, pd.DataFrame]
    ) -> ReporteValidacion:
        """Registra claves sin correspondencia; cada dirección es una regla distinta."""
        ambito = "relaciones_interim"
        faltantes = sorted(self.TABLAS_INTERIM - tablas.keys())

        if faltantes:
            return ReporteValidacion(
                ambito,
                (
                    RegistroValidacion(
                        ambito,
                        "tablas_requeridas",
                        "error",
                        f"Faltan tablas: {faltantes}",
                    ),
                ),
            )

        registros: list[RegistroValidacion] = []

        for origen, columna_origen, destino, columna_destino in self.REFERENCIAS_INTERIM:
            izquierda = tablas[origen]
            derecha = tablas[destino]
            regla = f"{origen}.{columna_origen} -> {destino}.{columna_destino}"

            if not isinstance(izquierda, pd.DataFrame) or not isinstance(
                derecha, pd.DataFrame
            ):
                registros.append(
                    RegistroValidacion(
                        ambito, regla, "error", "Se esperaban dos DataFrames"
                    )
                )
                continue

            if (
                columna_origen not in izquierda.columns
                or columna_destino not in derecha.columns
                or izquierda.columns.has_duplicates
                or derecha.columns.has_duplicates
            ):
                registros.append(
                    RegistroValidacion(
                        ambito,
                        regla,
                        "error",
                        "Falta alguna columna o hay columnas duplicadas",
                        columnas=(columna_origen, columna_destino),
                    )
                )
                continue

            sin_referencia = izquierda[columna_origen].isna() | (
                ~izquierda[columna_origen].isin(derecha[columna_destino].dropna())
            )
            indices = tuple(izquierda.index[sin_referencia].tolist())

            registros.append(
                RegistroValidacion(
                    tabla=ambito,
                    regla=regla,
                    estado="error" if indices else "ok",
                    detalle=(
                        f"{len(indices)} fila(s) de {origen} "
                        f"sin correspondencia en {destino}"
                        if indices
                        else "Todas las claves tienen correspondencia"
                    ),
                    filas=indices,
                    columnas=(columna_origen,) if indices else (),
                )
            )

        return ReporteValidacion(ambito, tuple(registros))

    def validar_consistencia_interim(
        self, tablas: Mapping[str, pd.DataFrame]
    ) -> ReporteValidacion:
        """Compara identidad y metadatos; presupone esquemas y claves ya validados."""
        ambito = "consistencia_interim"
        requeridas = {"diputados", "detalle_votaciones", "proyecto_ley"}
        faltantes = sorted(requeridas - tablas.keys())

        if faltantes:
            return ReporteValidacion(
                ambito,
                (
                    RegistroValidacion(
                        ambito,
                        "tablas_requeridas",
                        "error",
                        f"Faltan tablas: {faltantes}",
                    ),
                ),
            )

        diputados = tablas["diputados"]
        detalle = tablas["detalle_votaciones"]
        proyecto = tablas["proyecto_ley"]

        identidad = ("nombre", "apellido_paterno", "apellido_materno")
        campos_detalle = set(
            ("diputado_id", "votacion_id", *identidad, *self.CAMPOS_VOTACION)
        )
        campos_diputados = set(("diputado_id", *identidad))
        campos_proyecto = set(("Id", *self.EQUIVALENCIAS_PROYECTO))

        if (
            not all(
                isinstance(df, pd.DataFrame)
                for df in (diputados, detalle, proyecto)
            )
            or not campos_detalle.issubset(detalle.columns)
            or not campos_diputados.issubset(diputados.columns)
            or not campos_proyecto.issubset(proyecto.columns)
        ):
            return ReporteValidacion(
                ambito,
                (
                    RegistroValidacion(
                        ambito,
                        "esquemas",
                        "error",
                        "Faltan columnas necesarias o alguna entrada no es DataFrame",
                    ),
                ),
            )

        if (
            diputados["diputado_id"].duplicated().any()
            or proyecto["Id"].duplicated().any()
            or any(
                df.columns.has_duplicates
                for df in (diputados, detalle, proyecto)
            )
        ):
            return ReporteValidacion(
                ambito,
                (
                    RegistroValidacion(
                        ambito,
                        "claves",
                        "error",
                        "Hay claves o columnas duplicadas en las entradas",
                    ),
                ),
            )

        registros: list[RegistroValidacion] = []

        # Identidad del detalle frente a la tabla maestra.
        maestra = detalle[["diputado_id", *identidad]].merge(
            diputados[["diputado_id", *identidad]],
            on="diputado_id",
            how="left",
            suffixes=("_detalle", "_maestra"),
            indicator=True,
            validate="many_to_one",
        )

        diferencias = maestra["_merge"].eq("both")
        diferencias &= pd.concat(
            [
                maestra[f"{campo}_detalle"]
                .fillna("<NULO>")
                .ne(maestra[f"{campo}_maestra"].fillna("<NULO>"))
                for campo in identidad
            ],
            axis=1,
        ).any(axis=1)

        filas = tuple(detalle.index[diferencias].tolist())
        registros.append(
            RegistroValidacion(
                ambito,
                "identidad_diputado",
                "advertencia" if filas else "ok",
                f"{len(filas)} voto(s) con identidad distinta de la maestra",
                filas=filas,
                columnas=identidad if filas else (),
            )
        )

        # Cada campo debe ser constante dentro de una votación.
        for campo in self.CAMPOS_VOTACION:
            ids_inconsistentes = detalle.groupby(
                "votacion_id", dropna=False
            )[campo].nunique(dropna=False)
            ids_inconsistentes = ids_inconsistentes.index[
                ids_inconsistentes > 1
            ]

            filas = tuple(
                detalle.index[
                    detalle["votacion_id"].isin(ids_inconsistentes)
                ].tolist()
            )
            registros.append(
                RegistroValidacion(
                    ambito,
                    f"metadatos_constantes.{campo}",
                    "error" if filas else "ok",
                    f"{len(ids_inconsistentes)} votación(es) con {campo} variable",
                    filas=filas,
                    columnas=(campo,) if filas else (),
                )
            )

        # Catálogo frente a metadatos del detalle.
        resumen = detalle.groupby("votacion_id", as_index=False)[
            list(self.EQUIVALENCIAS_PROYECTO.values())
        ].first()

        comparacion = proyecto.merge(
            resumen,
            left_on="Id",
            right_on="votacion_id",
            how="left",
            indicator=True,
            validate="one_to_one",
        )

        ausentes = comparacion["_merge"].ne("both")
        if ausentes.any():
            registros.append(
                RegistroValidacion(
                    ambito,
                    "catalogo_sin_detalle",
                    "error",
                    f"{int(ausentes.sum())} votación(es) sin detalle para comparar",
                    filas=tuple(proyecto.index[ausentes].tolist()),
                )
            )

        for origen, destino in self.EQUIVALENCIAS_PROYECTO.items():
            a = comparacion[origen].astype("string").fillna("<NULO>")
            b = comparacion[destino].astype("string").fillna("<NULO>")

            if origen == "Descripcion":
                # F2_04 documenta diferencias de espacios entre ambas respuestas.
                a = a.str.replace(r"\s+", "", regex=True)
                b = b.str.replace(r"\s+", "", regex=True)
            else:
                a = a.str.strip()
                b = b.str.strip()

            diferentes = comparacion["_merge"].eq("both") & a.ne(b)
            filas = tuple(proyecto.index[diferentes].tolist())

            registros.append(
                RegistroValidacion(
                    ambito,
                    f"proyecto_vs_detalle.{origen}",
                    "error" if filas else "ok",
                    f"{len(filas)} votación(es) con {origen} distinto entre fuentes",
                    filas=filas,
                    columnas=(origen, destino) if filas else (),
                )
            )

        return ReporteValidacion(ambito, tuple(registros))
    
    def validar_periodos_y_totales_interim(
        self, tablas: Mapping[str, pd.DataFrame]
    ) -> ReporteValidacion:
        """Comprueba período único y votos nominales frente al catálogo oficial."""
        ambito = "periodos_y_totales_interim"
        obligatorias = {
            "periodos": {"fecha_inicio", "fecha_termino"},
            "proyecto_ley": {
                "Id",
                "Fecha",
                "TotalSi",
                "TotalNo",
                "TotalAbstencion",
                "TotalDispensado",
            },
            "detalle_votaciones": {"votacion_id", "opcion_voto"},
        }

        for nombre, columnas in obligatorias.items():
            df = tablas.get(nombre)
            if not isinstance(df, pd.DataFrame) or not columnas.issubset(df.columns):
                return ReporteValidacion(
                    ambito,
                    (
                        RegistroValidacion(
                            ambito,
                            "entradas",
                            "error",
                            f"Falta {nombre} o sus columnas: {sorted(columnas)}",
                        ),
                    ),
                )

        periodos = tablas["periodos"]
        proyecto = tablas["proyecto_ley"]
        detalle = tablas["detalle_votaciones"]

        fechas = pd.to_datetime(
            proyecto["Fecha"], errors="coerce", format="mixed"
        )
        inicios = pd.to_datetime(
            periodos["fecha_inicio"], errors="coerce", format="mixed"
        )
        terminos = pd.to_datetime(
            periodos["fecha_termino"], errors="coerce", format="mixed"
        )

        if fechas.isna().any() or inicios.isna().any() or terminos.isna().any():
            registro_periodos = RegistroValidacion(
                ambito,
                "periodo_unico",
                "error",
                "Hay fechas ausentes o inválidas",
                columnas=("Fecha", "fecha_inicio", "fecha_termino"),
            )
        else:
            coincidencias = pd.Series(0, index=proyecto.index)

            for inicio, termino in zip(inicios, terminos, strict=True):
                # Convención acordada en F3: [inicio, termino].
                coincidencias += (
                    (inicio <= fechas) & (fechas <= termino)
                ).astype(int)

            incorrectas = coincidencias.ne(1)
            registro_periodos = RegistroValidacion(
                ambito,
                "periodo_unico",
                "error" if incorrectas.any() else "ok",
                f"{int(incorrectas.sum())} votación(es) sin un período único",
                filas=tuple(proyecto.index[incorrectas].tolist()),
                columnas=("Fecha",) if incorrectas.any() else (),
            )

        totales = (
            "TotalSi",
            "TotalNo",
            "TotalAbstencion",
            "TotalDispensado",
        )
        oficiales = proyecto[list(totales)].apply(
            pd.to_numeric, errors="coerce"
        )

        if oficiales.isna().any().any() or (oficiales < 0).any().any():
            registro_totales = RegistroValidacion(
                ambito,
                "conciliacion_votos",
                "error",
                "Hay totales oficiales inválidos",
                columnas=totales,
            )
        else:
            nominales = detalle.groupby(
                ["votacion_id", "opcion_voto"]
            ).size()

            categorias = {
                "TotalSi": "Afirmativo",
                "TotalNo": "En Contra",
                "TotalAbstencion": "Abstención",
                "TotalDispensado": "Dispensado",
            }

            errores = []

            for posicion, (_, votacion) in enumerate(proyecto.iterrows()):
                identificador = votacion["Id"]

                observados = {
                    opcion: int(nominales.get((identificador, opcion), 0))
                    for opcion in categorias.values()
                }
                cantidad_total = int(
                    detalle["votacion_id"].eq(identificador).sum()
                )

                correcto = all(
                    observados[opcion] == int(oficiales.iloc[posicion][campo])
                    for campo, opcion in categorias.items()
                )
                correcto &= (
                    cantidad_total == int(oficiales.iloc[posicion].sum())
                )

                if not correcto:
                    errores.append(proyecto.index[posicion])

            registro_totales = RegistroValidacion(
                ambito,
                "conciliacion_votos",
                "error" if errores else "ok",
                (
                    f"{len(errores)} votación(es) con votos nominales "
                    "distintos de los totales"
                ),
                filas=tuple(errores),
                columnas=totales if errores else (),
            )

        return ReporteValidacion(
            ambito, (registro_periodos, registro_totales)
        )