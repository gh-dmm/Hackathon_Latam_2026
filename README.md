# Hackathon_Latam_2026
Hackathon Latinoamérica 2026 sobre desafíos del agua.

## Qué quedó completado
- Se integró una carga robusta de todos los archivos CSV/XLSX del proyecto.
- Se corrigieron los scripts de optimización genética, entrenamiento de RNA y proyección para que funcionen con los datos reales del repositorio.
- Se añadió un helper para leer datos con formatos heterogéneos y preparar ventanas semanales.
- Se validó el pipeline con pruebas automáticas.

## Cómo ejecutar
1. Instala las dependencias:
   - `python -m pip install pandas numpy scipy deap scikit-learn joblib matplotlib openpyxl pytest`
2. Desde la carpeta del proyecto, ejecuta:
   - `python hackatonlatam2026/genetico_v3.py`
   - `python hackatonlatam2026/entrenar_rna.py`
   - `python hackatonlatam2026/modelado1.py`
   - `python hackatonlatam2026/gui_app.py`

La interfaz nueva te pedirá:
- una fecha inicial seleccionada desde las fechas disponibles en los datasets,
- cuántas semanas considerar (7, 26 o 52),
- y mostrará una gráfica comparando la optimización genética contra los datos originales, además de una segunda gráfica con la mejora genética.

## Archivos principales
- [hackatonlatam2026/genetico_v3.py](hackatonlatam2026/genetico_v3.py)
- [hackatonlatam2026/entrenar_rna.py](hackatonlatam2026/entrenar_rna.py)
- [hackatonlatam2026/modelado1.py](hackatonlatam2026/modelado1.py)
- [hackatonlatam2026/data_utils.py](hackatonlatam2026/data_utils.py)
