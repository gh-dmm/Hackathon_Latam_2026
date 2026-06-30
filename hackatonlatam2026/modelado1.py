import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN INICIAL
# Modifica 'S_inicial' con el valor de volumen real que obtengas del embalse.
# ==============================================================================
S_inicial = 1000000
V_max = 3387000 # Volumen máximo del embalse (puedes ajustarlo según tus necesidades) 
u_t = 0 # Extracción diaria (puedes ajustarla según tus necesidades)
S_min = 0.25 * V_max # El 25% del volumen total
# ==============================================================================

# 1. Cargar datasets
df_storage = pd.read_csv('DataSetExport-Discharge Total.Change-in-Storage@08461200-Instantaneous-TCM-20260622185956.csv', skiprows=1)
# El archivo de elevación usa ';' como separador
# Cambia esta línea en tu archivo modelado1.py:
df_elev = pd.read_csv('DataSetExport-Reservoir Elevation.Web-Daily-m@08461200-Aggregate-m-20260629164752.csv', sep=';', skiprows=1)
# 2. Limpieza y preparación
df_storage.rename(columns={'Timestamp (UTC-06:00)': 't', 'Value (TCM)': 'Delta_S_obs'}, inplace=True)
df_storage['t'] = pd.to_datetime(df_storage['t'], errors='coerce')
df_storage = df_storage.dropna(subset=['t'])

df_elev.rename(columns={'Inicio de intervalo (UTC-06:00)': 't', 'Valor (m)': 'Elevacion'}, inplace=True)
# Convertimos elevación reemplazando ',' por '.' para que Python reconozca el número
df_elev['Elevacion'] = df_elev['Elevacion'].replace({',': '.'}, regex=True).astype(float)
df_elev['t'] = pd.to_datetime(df_elev['t'], errors='coerce')
df_elev = df_elev.dropna(subset=['t'])

# 3. Sincronización temporal: unimos sobre el rango donde hay datos de elevación
df = pd.merge(df_storage, df_elev, on='t', how='inner')

# 4. Cálculo del Balance de Masa acumulado
df['S_opt'] = S_inicial + (df['Delta_S_obs'] - u_t).cumsum()

# Asignamos u_t como columna para poder calcular las nuevas métricas (C_dev y C_smooth)
df['u_t'] = u_t

# Cálculo del costo de penalización cuadrático
# max(0, S_min - S_opt) es el déficit: si S_opt es mayor a S_min, el déficit es 0
df['deficit'] = (S_min - df['S_opt']).clip(lower=0)
df['costo_cuadratico'] = df['deficit']**2

# Costo total crítico
C_crit = df['costo_cuadratico'].sum()

# Cálculo de C_dev: Suma de los cuadrados de la demanda u(t)
C_dev = (df['u_t']**2).sum()

# Cálculo de C_smooth: Suma de los cuadrados de la diferencia entre extracciones sucesivas [u(t) - u(t-1)]^2
C_smooth = (df['u_t'].diff().fillna(0)**2).sum()

print(f"El Costo Crítico (C_crit) para un S_min de {S_min} TCM es: {C_crit:,.2f}")
# 5. Visualización
# 6. Mostrar variables en terminal
# Seleccionamos las columnas relevantes para inspección
df_resumen = df[['t', 'Delta_S_obs', 'u_t', 'S_opt', 'deficit', 'Elevacion']]

print("--- Resumen de variables (Primeras 10 filas) ---")
print(df_resumen.head(10))

print("\n--- Resumen de variables (Últimas 10 filas) ---")
print(df_resumen.tail(10))

print(f"\n--- Estadísticas generales ---")
print(f"Capacidad Máxima (V_max): {V_max:,.2f} TCM")
print(f"Nivel Crítico (S_min): {S_min:,.2f} TCM")
print(f"Costo Crítico Total (C_crit): {C_crit:,.2f}")
print(f"Desviación de Demanda (C_dev): {C_dev:,.2f}")
print(f"Suavidad de Operación (C_smooth): {C_smooth:,.2f}")

fig, ax1 = plt.subplots(figsize=(14, 7))

# Eje izquierdo para Volumen
ax1.plot(df['t'], df['S_opt'], color='blue', label='Almacenamiento $S_{opt}$ (TCM)')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Volumen (TCM)', color='blue')

# Eje derecho para Elevación
ax2 = ax1.twinx()
ax2.plot(df['t'], df['Elevacion'], color='green', linestyle=':', label='Elevación (m)')
ax2.set_ylabel('Elevación (m)', color='green')

plt.title('Comportamiento Hidrológico: S_opt vs Elevación Real')
ax1.grid(True, linestyle='--')
# En la parte de graficación:
ax1.axhline(y=S_min, color='red', linestyle='--', label=f'S_min (25% = {S_min:,.0f} TCM)')
ax1.legend(loc='upper left')
plt.show()