import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL
# ==============================================================================
S_inicial = 1000000
V_max = 3387000 # Volumen máximo del embalse 
u_t_fija = 0 # Extracción diaria base
S_min = 0.25 * V_max # El 25% del volumen total

# Decisiones del Sistema de Control (RNA)
diccionario_decisiones = {0: 0, 1: 2500, 2: 5000}
# ==============================================================================

# ==============================================================================
# 2. CARGA Y LIMPIEZA DE DATASETS
# ==============================================================================
df_storage = pd.read_csv('DataSetExport-Discharge Total.Change-in-Storage@08461200-Instantaneous-TCM-20260622185956.csv', skiprows=1)
df_elev = pd.read_csv(r'C:\Users\Usuario\Downloads\DataSetExport-Reservoir Elevation.Web-Daily-m@08461200-Aggregate-m-20260629164752.csv', sep=';', skiprows=1)

df_storage.rename(columns={'Timestamp (UTC-06:00)': 't', 'Value (TCM)': 'Delta_S_obs'}, inplace=True)
df_storage['t'] = pd.to_datetime(df_storage['t'], errors='coerce')
df_storage = df_storage.dropna(subset=['t'])

df_elev.rename(columns={'Inicio de intervalo (UTC-06:00)': 't', 'Valor (m)': 'Elevacion'}, inplace=True)
df_elev['Elevacion'] = df_elev['Elevacion'].replace({',': '.'}, regex=True).astype(float)
df_elev['t'] = pd.to_datetime(df_elev['t'], errors='coerce')
df_elev = df_elev.dropna(subset=['t'])

# Sincronización temporal: unimos sobre el rango donde hay datos de elevación
df = pd.merge(df_storage, df_elev, on='t', how='inner')

# ==============================================================================
# 3. BALANCE DE MASA (HISTÓRICO)
# ==============================================================================
df['u_t'] = u_t_fija
df['S_opt'] = S_inicial + (df['Delta_S_obs'] - df['u_t']).cumsum()
df['deficit'] = (S_min - df['S_opt']).clip(lower=0)
df['costo_cuadratico'] = df['deficit']**2

# ==============================================================================
# 4. RED NEURONAL ARTIFICIAL (RNA) - PROYECCIÓN 26 SEMANAS
# ==============================================================================
print("\n--- Entrenando Red Neuronal Artificial (RNA) ---")
condiciones = [
    (df['Delta_S_obs'] < -1000),
    (df['Delta_S_obs'] >= -1000) & (df['Delta_S_obs'] <= 1000),
    (df['Delta_S_obs'] > 1000)
]
opciones_entrenamiento = [0, 1, 2]
df['Decision_Historica'] = np.select(condiciones, opciones_entrenamiento, default=1)

X_train = df[['Delta_S_obs', 'Elevacion']] 
y_train = df['Decision_Historica']         

rna = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
rna.fit(X_train, y_train)

dias_proyeccion = 26 * 7
fecha_ultima = df['t'].max()
fechas_futuras = pd.date_range(start=fecha_ultima + pd.Timedelta(days=1), periods=dias_proyeccion, freq='D')

mean_delta = df['Delta_S_obs'].mean()
std_delta = df['Delta_S_obs'].std()
mean_elev = df['Elevacion'].mean()

sim_delta_s = np.random.normal(mean_delta, std_delta, dias_proyeccion)
sim_elev = np.random.normal(mean_elev, 0.5, dias_proyeccion) 

df_futuro = pd.DataFrame({'t': fechas_futuras, 'Delta_S_obs': sim_delta_s, 'Elevacion': sim_elev})
X_future = df_futuro[['Delta_S_obs', 'Elevacion']]
df_futuro['Decision_RNA'] = rna.predict(X_future)
df_futuro['u_t'] = df_futuro['Decision_RNA'].map(diccionario_decisiones)

ultimo_S_opt = df['S_opt'].iloc[-1]
df_futuro['S_opt'] = ultimo_S_opt + (df_futuro['Delta_S_obs'] - df_futuro['u_t']).cumsum()
df_futuro['deficit'] = (S_min - df_futuro['S_opt']).clip(lower=0)
df_futuro['costo_cuadratico'] = df_futuro['deficit']**2

# Fusión del Histórico y la Proyección
df_total = pd.concat([df, df_futuro], ignore_index=True)

# ==============================================================================
# 5. CÁLCULO DE MÉTRICAS DE CONTROL
# ==============================================================================
C_crit = df_total['costo_cuadratico'].sum()
C_dev = (df_total['u_t']**2).sum()
C_smooth = (df_total['u_t'].diff().fillna(0)**2).sum()

df_resumen = df_total[['t', 'Delta_S_obs', 'u_t', 'S_opt', 'deficit', 'Elevacion']]
print("\n--- Resumen de variables (Primeras 10 filas) ---")
print(df_resumen.head(10))
print("\n--- Resumen de variables (Últimas 10 filas proyectadas por RNA) ---")
print(df_resumen.tail(10))

print(f"\n--- Estadísticas generales ---")
print(f"Capacidad Máxima (V_max): {V_max:,.2f} TCM")
print(f"Nivel Crítico (S_min): {S_min:,.2f} TCM")
print(f"Costo Crítico Total (C_crit): {C_crit:,.2f}")
print(f"Desviación de Demanda (C_dev): {C_dev:,.2f}")
print(f"Suavidad de Operación (C_smooth): {C_smooth:,.2f}")

# ==============================================================================
# 6. SISTEMA DE CONTROL CUÁNTICO DINÁMICO
# ==============================================================================
print("\n--- SISTEMA DE CONTROL CUÁNTICO INICIADO ---")
# Vector de estado inicial |0>
psi_0 = np.array([[1], [0]])
# Matriz unitaria U (Hadamard)
H_unitaria = (1 / np.sqrt(2)) * np.array([[1,  1], [1, -1]])

def evolucion_cuantica_por_dataset(delta_s, psi_actual):
    factor = delta_s / 100000.0 # Escalador para manejar amplitudes cuánticas
    beta_t = np.array([[0, -1j * factor], [1j * factor, 0]])
    sistema_control = np.matmul(H_unitaria, beta_t)
    psi_nuevo = np.matmul(sistema_control, psi_actual)
    return psi_nuevo, beta_t

print("Vector de Estado Inicial |0>:")
print(psi_0)
psi_actual = psi_0

# Iteramos sobre los primeros 5 días históricos para demostrar la evolución del tensor
for index, row in df.head(5).iterrows():
    fecha = row['t'].strftime('%Y-%m-%d')
    delta_s = row['Delta_S_obs']
    psi_actual, beta_aplicada = evolucion_cuantica_por_dataset(delta_s, psi_actual)
    
    print(f"\n--- Día {fecha} | Delta S real: {delta_s:,.2f} TCM ---")
    print("Matriz de Control Beta(t):")
    print(np.round(beta_aplicada, 5))
    print("Nuevo Vector de Estado Psi(t):")
    print(np.round(psi_actual, 5))

# ==============================================================================
# 7. VISUALIZACIÓN MEGA PRO
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(df['t'], df['S_opt'], color='blue', label='Histórico $S_{opt}$ (TCM)')
ax1.plot(df_futuro['t'], df_futuro['S_opt'], color='purple', linestyle='--', label='Proyección RNA (26 Semanas)')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Volumen (TCM)', color='blue')

ax2 = ax1.twinx()
ax2.plot(df['t'], df['Elevacion'], color='green', linestyle=':', label='Elevación Histórica (m)')
ax2.set_ylabel('Elevación (m)', color='green')

ax1.axvline(x=fecha_ultima, color='black', linestyle='-.', label='Inicio de Proyección')
ax1.axhline(y=S_min, color='red', linestyle='--', label=f'S_min (25% = {S_min:,.0f} TCM)')

plt.title('Integración Total: Balance Hidrológico + Control RNA')
ax1.grid(True, linestyle='--')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()
