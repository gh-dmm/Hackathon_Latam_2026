import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL
# ==============================================================================
S_inicial = 1000000 
V_max = 3387000
S_min = 0.25 * V_max

# ==============================================================================
# 2. CARGAR MODELO Y ESTADÍSTICAS HISTÓRICAS
# ==============================================================================
print("--- Cargando Modelo de Control RNA (Basado en Genético) ---")
try:
    rna = joblib.load('modelo_rna_genetico.pkl')
except FileNotFoundError:
    print("Error: Ejecuta 'entrenar_rna_genetico.py' primero.")
    exit()

# Para proyectar el futuro necesitamos la media y desviación del Delta S histórico
# Cargamos un resumen rápido de los datos
df_cambio = pd.read_csv("Cambio_almacenamiento_historico.csv")
df_cambio.columns = df_cambio.columns.str.strip()
if 'Timestamp (UTC-06:00)' in df.columns:
    df_cambio.rename(columns={'Timestamp (UTC-06:00)': 'Fecha'}, inplace=True)
df_cambio['Fecha'] = pd.to_datetime(df_cambio['Fecha'], errors='coerce')
cambio_semanal = df_cambio.set_index('Fecha')['Value (TCM)'].resample('W-SUN').sum()

mean_delta = cambio_semanal.mean()
std_delta = cambio_semanal.std()

# ==============================================================================
# 3. PROYECCIÓN 26 SEMANAS (SIMULACIÓN DINÁMICA)
# ==============================================================================
semanas_proyeccion = 26
fecha_inicio_proyeccion = pd.Timestamp.today().normalize()
fechas_futuras = pd.date_range(start=fecha_inicio_proyeccion, periods=semanas_proyeccion, freq='W-SUN')

# Generamos escenarios aleatorios de entrada de agua (Monte Carlo estadístico)
sim_delta_s = np.random.normal(mean_delta, std_delta, semanas_proyeccion)

resultados_futuros = []
s_opt_actual = S_inicial

print("\n--- Iniciando Simulación de 26 Semanas ---")
for t in range(semanas_proyeccion):
    delta_s_t = sim_delta_s[t]
    
    # 1. La RNA observa el estado actual (Delta S de la semana y Nivel de Presa)
    estado_actual = pd.DataFrame({'Delta_S_obs': [delta_s_t], 'S_actual': [s_opt_actual]})
    
    # 2. La RNA toma la decisión basada en lo que aprendió del Genético
    decision_str = rna.predict(estado_actual)[0]
    u_t_aplicado = float(decision_str)
    
    # 3. Aplicamos la decisión y actualizamos el embalse (Balance de masa)
    s_opt_nuevo = s_opt_actual + delta_s_t - u_t_aplicado
    
    # Restricción física (el agua no puede ser negativa ni superar V_max)
    s_opt_nuevo = max(0, min(s_opt_nuevo, V_max))
    
    resultados_futuros.append({
        'Fecha': fechas_futuras[t],
        'Delta_S_Simulado': delta_s_t,
        'Decision_RNA_u': u_t_aplicado,
        'S_opt_Proyectado': s_opt_nuevo
    })
    
    # Actualizamos la variable para el siguiente ciclo
    s_opt_actual = s_opt_nuevo

df_futuro = pd.DataFrame(resultados_futuros)

# ==============================================================================
# 4. MÉTRICAS DE RESILIENCIA Y RESULTADOS
# ==============================================================================
df_futuro['Deficit'] = (S_min - df_futuro['S_opt_Proyectado']).clip(lower=0)
C_crit = (df_futuro['Deficit']**2).sum()
C_dev = (df_futuro['Decision_RNA_u']**2).sum()

print("\nPrimeras 5 semanas de proyección:")
print(df_futuro[['Fecha', 'Delta_S_Simulado', 'Decision_RNA_u', 'S_opt_Proyectado']].head(5))

print(f"\n--- Estadísticas de la Proyección (26 Semanas) ---")
print(f"Costo Crítico Total (Sequía): {C_crit:,.2f}")
print(f"Desviación de Operación Acumulada: {df_futuro['Decision_RNA_u'].sum():,.2f} TCM")

# ==============================================================================
# 5. VISUALIZACIÓN
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df_futuro['Fecha'], df_futuro['S_opt_Proyectado'], color='purple', marker='o', label='S_opt (Proyección RNA Genética)')
ax1.axhline(y=S_min, color='red', linestyle='--', label=f'Nivel Crítico S_min ({S_min:,.0f} TCM)')
ax1.axhline(y=V_max, color='blue', linestyle=':', label='Capacidad Máxima')

ax1.set_xlabel('Fecha Proyectada')
ax1.set_ylabel('Volumen (TCM)')
plt.title('Proyección Autónoma a 26 Semanas: RNA Entrenada con Algoritmo Genético')
ax1.grid(True, linestyle='--')
ax1.legend(loc='upper right')

plt.tight_layout()
plt.show()
