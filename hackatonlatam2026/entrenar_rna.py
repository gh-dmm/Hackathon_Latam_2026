import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from genetico_v3 import preparar_ventana_semanal, mutar_nivel
from sklearn.neural_network import MLPClassifier
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. CARGA Y PREPARACIÓN SEMANAL (Igual al Genético)
# =========================================================
print("Cargando y procesando datos históricos...")
df_lib = pd.read_excel("R_observ.xlsx")
df_cambio = pd.read_csv("Cambio_almacenamiento_historico.csv")
df_total = pd.read_csv("DataSetExport-Total Storage.csv")

def limpiar_dataset(df):
    df.columns = df.columns.str.strip()
    if 'Timestamp (UTC-06:00)' in df.columns:
        df.rename(columns={'Timestamp (UTC-06:00)': 'Fecha'}, inplace=True)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha']).set_index('Fecha')
    return df

df_lib = limpiar_dataset(df_lib)
df_cambio = limpiar_dataset(df_cambio)
df_total = limpiar_dataset(df_total)

# Agrupación Semanal
lib_semanal = df_lib['Valor'].resample('W-SUN').mean() * 604.8
cambio_semanal = df_cambio['Value (TCM)'].resample('W-SUN').sum()
total_semanal = df_total['Value (TCM)'].resample('W-SUN').first()

# Consolidar en un solo DataFrame histórico
df_hist = pd.DataFrame({
    'R_obs': lib_semanal,
    'Delta_S_obs': cambio_semanal,
    'S_actual': total_semanal
}).dropna()

# Usaremos las últimas 104 semanas (2 años) para generar datos de entrenamiento estables
df_hist = df_hist.tail(104)
T = len(df_hist)

R_obs = df_hist['R_obs'].values
Delta_S_obs = df_hist['Delta_S_obs'].values
S_inicial = df_hist['S_actual'].iloc[0]

# Parámetros físicos
S_max = 3387000.0
S_min = 0.25 * S_max
mediana_R = np.median(R_obs)
delta_u = 0.25 * mediana_R
u_max = 2 * delta_u
NIVELES_PERMITIDOS = [-2*delta_u, -delta_u, 0.0, delta_u, 2*delta_u]

# =========================================================
# 2. ALGORITMO GENÉTICO (Generador de Etiquetas)
# =========================================================
print(f"\nIniciando Algoritmo Genético para optimizar {T} semanas históricas...")
if 'FitnessMin' in creator.__dict__: del creator.FitnessMin
if 'Individual' in creator.__dict__: del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_level", random.choice, NIVELES_PERMITIDOS)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_level, n=T)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluar_srs(individuo):
    cromosoma_u = np.array(individuo)
    S_opt = np.zeros(T + 1)
    S_opt[0] = S_inicial
    
    if abs(np.sum(cromosoma_u)) > 0.10 * np.sum(R_obs): return (float('1e10'),)
    
    for t in range(T):
        S_opt[t+1] = S_opt[t] + Delta_S_obs[t] - cromosoma_u[t]
        if R_obs[t] + cromosoma_u[t] < 0: return (float('1e10'),)
        if S_opt[t+1] < 0 or S_opt[t+1] > S_max: return (float('1e10'),)

    C_crit = np.sum([max(0, S_min - S_opt[t])**2 for t in range(T+1)])
    C_dev = np.sum(cromosoma_u**2)
    C_smooth = np.sum(np.diff(cromosoma_u)**2)

    w1, w2, w3 = 1/((T+1)*(S_min**2)), 0.1/(T*(u_max**2)), 0.1/((T-1)*((2*u_max)**2))
    return ((w1 * C_crit) + (w2 * C_dev) + (w3 * C_smooth),)

toolbox.register("evaluate", evaluar_srs)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", lambda ind, indpb: ([(random.choice(NIVELES_PERMITIDOS) if random.random() < indpb else x) for x in ind],), indpb=0.15)
toolbox.register("select", tools.selTournament, tournsize=4)

pop = toolbox.population(n=300)
pop[0] = creator.Individual([0.0] * T) # Línea base
hof = tools.HallOfFame(1)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, halloffame=hof, verbose=False)

mejor_secuencia = np.array(hof[0])
df_hist['Decision_Optima_u'] = mejor_secuencia
print("¡Genético terminado! Secuencia óptima encontrada.")

# =========================================================
# 3. ENTRENAMIENTO DE LA RED NEURONAL (Imitation Learning)
# =========================================================
print("\nEntrenando la Red Neuronal para imitar al Genético...")
# La RNA usará el Cambio de Almacenamiento y el Almacenamiento Actual para decidir
X_train = df_hist[['Delta_S_obs', 'S_actual']]
# La etiqueta a predecir es la decisión óptima del genético
y_train = df_hist['Decision_Optima_u']

# Convertimos las decisiones a variables categóricas (str) para clasificación segura
y_train_cat = y_train.astype(str)

rna = MLPClassifier(hidden_layer_sizes=(15, 10), max_iter=1500, random_state=42)
rna.fit(X_train, y_train_cat)

joblib.dump(rna, 'modelo_rna_genetico.pkl')
print("\n¡Éxito! Modelo RNA entrenado y guardado como 'modelo_rna_genetico.pkl'.")

plt.plot(rna.loss_curve_, color='teal')
plt.title('Curva de Aprendizaje de la RNA (Imitando al Genético)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.grid(True, linestyle='--')
plt.show()
