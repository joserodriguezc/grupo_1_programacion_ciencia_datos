"""Columnas de los cinco CSV interim"""

COLUMNAS_VOTACION_PROYECTO = (
    "numero_boletin",
    "Id",
    "Descripcion",
    "Fecha",
    "TotalSi",
    "TotalNo",
    "TotalAbstencion",
    "TotalDispensado",
    "Quorum",
    "Resultado",
    "Tipo",
    "TipoVotacionProyectoLey",
    "Articulo",
    "TramiteConstitucional",
    "TramiteReglamentario",
)

COLUMNAS_PERIODO_LEGISLATIVO = (
    "periodo_id",
    "nombre",
    "fecha_inicio",
    "fecha_termino",
)

COLUMNAS_DIPUTADO = (
    "diputado_id",
    "nombre",
    "nombre2",
    "apellido_paterno",
    "apellido_materno",
    "fecha_nacimiento",
    "rut",
    "rut_dv",
    "sexo_valor",
    "sexo_desc",
    "periodo_id",
    "fecha_inicio_periodo",
    "fecha_termino_periodo",
)

COLUMNAS_MILITANCIA = (
    "diputado_id",
    "partido_id",
    "partido_nombre",
    "partido_alias",
    "fecha_inicio",
    "fecha_termino",
)

COLUMNAS_VOTO_NOMINAL = (
    "diputado_id",
    "nombre",
    "nombre2",
    "apellido_paterno",
    "apellido_materno",
    "opcion_codigo",
    "opcion_voto",
    "votacion_id",
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
