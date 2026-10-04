# Convenciones de los notebooks de F4

Este documento registra la revisión de coherencia de los notebooks F4_01 a F4_05 y fija las
convenciones de estructura y visualización que deben seguir todos, incluido F4_06.
Las convenciones visuales están implementadas en `F4/src/analisis/visualizacion.py`.

## 1. Revisión

### Estructura

| Aspecto | Antes | Ahora |
|---|---|---|
| Portada | F4_03 sin "Entradas"; F4_04 con "Resumen ejecutivo" y "Contenido" | Las cinco con `# F4 · …`, Objetivo, Entradas y organización, Cautelas |
| Detección de la raíz | Cinco variantes distintas | Una sola: `pyproject.toml` + `F4/src/analisis` |
| Imports | F4_02 usaba `analisis.…` con otra ruta en `sys.path` | Todos `from F4.src.analisis…` |
| Validación contra lo versionado | F4_03 no validaba | F4_03 compara con `universo_por_metodo.csv` |
| Reproducibilidad | F4_03 sin sección; F4_02 dentro de Exportación | Sección propia en los cinco, con figuras generadas |
| Síntesis | F4_04 sin síntesis | Los cinco terminan con Síntesis |
| Numeración | F4_03 con una "3.3" después de la 4.2 y un gráfico repetido | Secciones consecutivas, sin duplicados |

### Visualización

| Problema | Dónde | Corrección |
|---|---|---|
| El azul significaba "Sí" y también "lado R" | F4_01, F4_05 frente a F4_02–F4_04 | Votos con paleta propia (verde, morado, gris) |
| Valores por defecto de matplotlib, sin nota de fuente | F4_03 | Estilo común, títulos a la izquierda, notas |
| Leyenda invertida (P_p < 0 en azul) y texto que la contradecía | F4_03 | Rojo = lado L, azul = lado R, igual que B-Call |
| `b_pj` (con signo) en `viridis`; Hamming (0–1) en `coolwarm` | F4_03 | Divergente rojo-azul centrada en 0; secuencial para magnitudes |
| Matriz de Hamming con partido "modal" (contradice el protocolo) | F4_03 | Solo la matriz con el partido vigente en cada votación |
| Tres variantes casi iguales de Hamming según co-votos | F4_03 | Un gráfico por categorías con mediana y rango intercuartílico |
| Gráfico de "cambio de P_p" que repetía el de P_p | F4_03 | Gráfico del cambio máximo y mediano por partido |
| Paleta propia, "Figura N." en los títulos, figura re-cargada con `imread` | F4_04 | Estilo común; la figura PC1–d1 se dibuja directamente |
| 160, 170 y 180 dpi según el notebook | Todos | 160 dpi, recorte ajustado, sin metadatos variables |

### Portabilidad (Windows)

- `.gitattributes` fija saltos de línea LF en `F2/data`, `F3/data` y `F4/data`. Los archivos de F2
  versionados con CRLF se conservan tal cual (`-text`).
- Todos los módulos y notebooks escriben CSV con `lineterminator='\n'` y JSON con `newline='\n'`.
  Así los hashes de manifiestos, auditoría y conciliación son iguales en cualquier sistema.

## 2. Estructura de un notebook

1. **Portada** (`# F4 · <tema>`) con `## Objetivo`, `## Entradas y organización` y
   `### Cautelas de interpretación`.
2. **Celda de preparación**: imports estándar; raíz detectada por `pyproject.toml` y
   `F4/src/analisis`; imports `from F4.src.analisis…`; diccionario `rutas` que valida que cada
   entrada exista y no esté vacía; lectura de `F4/config/analisis.toml`; `vis.aplicar_estilo()`;
   función `guardar(fig, nombre)` que registra las figuras generadas.
3. **Secciones numeradas** `## N.` y `### N.M.`, consecutivas.
4. **Validación**: todo resultado que el notebook recalcula se compara con su salida
   versionada, se imprime `Coincide con …: True` y se verifica con `assert`.
5. **Sensibilidad**: si el notebook estima o resume un método, muestra su sensibilidad de
   `sensibilidad.csv` (o del módulo correspondiente) con un gráfico, no solo con tablas.
6. **`## Reproducibilidad`** (penúltima): commit, archivos modificados, entorno, commit del corte
   F3, semilla y figuras generadas. Termina con `print('F4_0X_APROBADO=True')`, que
   `F4/test/notebooks/test_notebooks_f4.py` usa como indicador.
7. **`## Síntesis`** (última). Las referencias, si las hay, van después como anexo.

Los notebooks no reimplementan fórmulas: llaman a los módulos de `F4/src/analisis/`. Las salidas
de un módulo se guardan con su propia función, en su ubicación oficial, sin copias.

Las rutas, columnas y códigos de razón de cada salida están en `contratos.py` (A04). La cadena
de módulos se regenera con `python -m F4.src.analisis.pipeline` (D01), que termina con el
manifiesto de entrega (`exportacion.py`, F04). Las salidas de B-Call las produce F4_02.

F4_06 (comunicación) no estima nada: integra las salidas de F4_01 a F4_05 para responder las
preguntas de investigación. Cada cifra de su texto se calcula desde los archivos versionados
(`display(Markdown(...))`) y no se escribe a mano. El plano d1/d2 que comparte con F4_02 está en
`visualizacion.plano_bcall`.

## 3. Convenciones visuales

### Uso del módulo

```python
from F4.src.analisis import visualizacion as vis

vis.aplicar_estilo()
fig, ax = plt.subplots(figsize=(9, 5))
...
vis.titulo(ax, 'Título en negrita', 'subtítulo con el universo o la lectura')
vis.formato_coma(ax, 'x')              # coma decimal y signo menos tipográfico
vis.leyenda_inferior(ax, ncol=2)       # leyenda bajo el eje, sin marco
vis.nota(fig, 'Fuente: … ')            # fuente o lectura en gris
guardar(fig, 'nombre.png')             # 160 dpi, sin metadatos
```

### Paletas con significado fijo

Cada color tiene un único significado en todo F4. Las paletas categóricas se validaron para
daltonismo.

| Paleta | Valores | Uso |
|---|---|---|
| `GRUPOS` | L `#E74C3C` · R `#3498DB` · independientes `#5E5E5E` · L/R distintos `#D4A017` | Lado del eje B-Call y grupos externos; con forma de marcador (círculo, cuadrado, cruz, rombo) |
| `VOTOS` | Sí `#1a9850` · No `#762a83` · Abstención `#a3a3a3` · sin registro `#d9d8d4` | Decisión nominal o ternaria |
| `ESTADOS` / `ESTADOS_TEXTO` | ok · advertencia · error | Controles; siempre con la etiqueta escrita |
| `CLUSTERS` | grises `#3d3c39` · `#b5b3ad` | Clusters 1..k: etiquetas neutrales, sin rojo ni azul |
| `NEUTRO` | `#7a7974` | Una sola serie sin significado de grupo |

Los índices de cohesión no se codifican por color: cada panel los identifica por su título.

### Escalas de color

| Dato | Escala |
|---|---|
| Magnitud de 0 a 1 o conteos | `vis.SECUENCIAL` (un solo tono, de claro a oscuro) |
| Magnitud con signo sobre el eje L/R (`b_pj`, votos orientados) | `vis.DIVERGENTE_LR` con `vis.norma_simetrica(...)`, centrada en 0 |
| Voto ternario | `vis.VOTO_TERNARIO` con `vis.norma_ternaria()` |
| Celda calculada pero no publicable | rayado con borde `vis.RAYADO` |

### Reglas de composición

- Título en negrita alineado a la izquierda; sin "Figura N." dentro del título.
- En F4_06, el título dice el hallazgo, no el tema ("9 partidos forman un bloque compacto en el
  lado L", no "Posición partidaria"), con cifras calculadas desde los datos.
- En F4_06, bajo cada figura van cuatro frases: qué muestra, qué se infiere, qué límite tiene
  y cómo aporta al relato (función `lectura`). Están pensadas para copiarse al informe.
- Ejes con coma decimal; el texto siempre en tinta (`TEXTO`, `TEXTO_SECUNDARIO`), nunca en el
  color de una serie.
- Categorías discretas (votaciones, co-votos, escenarios): puntos o barras, no líneas que sugieran
  continuidad.
- Sobretrazado: burbujas por coordenada exacta (área proporcional) o desplazamiento con la semilla
  del protocolo (`reproducibilidad.semilla`), declarado en la nota.
- Leyendas con marcadores de tamaño fijo; el área de una burbuja se explica en la leyenda o en el
  subtítulo.

## 4. Lista de verificación para un notebook nuevo

- [ ] Portada con Objetivo, Entradas y organización, Cautelas.
- [ ] Celda de preparación estándar y `vis.aplicar_estilo()`.
- [ ] Ningún color hexadecimal escrito a mano: usar las paletas de `visualizacion.py`.
- [ ] Cada figura con título, ejes rotulados, coma decimal y nota de fuente cuando corresponda.
- [ ] Escrituras con `lineterminator='\n'` (CSV) y `newline='\n'` (texto).
- [ ] Comparación con `assert` contra las salidas versionadas que el notebook recalcula.
- [ ] Gráfico de sensibilidad si el notebook estima o resume un método.
- [ ] Secciones Reproducibilidad y Síntesis al final; indicador `F4_0X_APROBADO=True`.
- [ ] Notebook agregado a `F4/test/notebooks/test_notebooks_f4.py`.
- [ ] Ejecución completa sin errores y `ruff check` sin avisos nuevos.
