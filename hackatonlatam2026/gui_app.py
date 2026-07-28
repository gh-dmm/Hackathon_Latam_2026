from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from .data_utils import cargar_datos_hidrologicos
from .genetico_v3 import preparar_ventana_semanal


def obtener_fechas_disponibles(base_dir: Path | None = None) -> List[pd.Timestamp]:
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    datos = cargar_datos_hidrologicos(base_dir)
    dfs = [datos["lib"], datos["cambio"], datos["total"], datos["evap"]]
    fechas = []
    for df in dfs:
        if isinstance(df.index, pd.DatetimeIndex):
            fechas.extend(df.index.tolist())
    if not fechas:
        return []
    return sorted({f.normalize() for f in fechas if pd.notna(f)})


def construir_resultado_ventana(base_dir: Path | None = None, fecha_inicio: str = "2024-01-01", semanas: int = 26) -> Dict[str, Any]:
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    datos = cargar_datos_hidrologicos(base_dir)
    df_lib = datos["lib"]
    df_cambio = datos["cambio"]
    df_total = datos["total"]
    df_evap = datos["evap"]
    df_batimetria = datos["batimetria"]

    R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
        df_lib, df_cambio, df_total, df_evap, df_batimetria, fecha_inicio, semanas
    )

    S_max = 3387000.0
    S_min = 0.25 * S_max
    mediana_R = np.median(R_obs)
    delta_u = 0.25 * mediana_R
    u_max = 2 * delta_u
    niveles = [-2 * delta_u, -delta_u, 0.0, delta_u, 2 * delta_u]

    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_level", np.random.choice, niveles)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_level, n=semanas)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluar_srs(individuo):
        cromosoma_u = np.array(individuo)
        S_opt = np.zeros(semanas + 1)
        S_opt[0] = S_inicial
        if abs(np.sum(cromosoma_u)) > 0.10 * np.sum(R_obs):
            return (float("1e10"),)
        for t in range(semanas):
            S_opt[t + 1] = S_opt[t] + Delta_S_obs[t] - cromosoma_u[t]
            if R_obs[t] + cromosoma_u[t] < 0:
                return (float("1e10"),)
            if S_opt[t + 1] < 0 or S_opt[t + 1] > S_max:
                return (float("1e10"),)
        C_crit = np.sum([max(0, S_min - S_opt[t]) ** 2 for t in range(semanas + 1)])
        C_dev = np.sum(cromosoma_u ** 2)
        C_smooth = np.sum(np.diff(cromosoma_u) ** 2)
        w1 = 1 / ((semanas + 1) * (S_min ** 2))
        w2 = 0.1 / (semanas * (u_max ** 2))
        w3 = 0.1 / ((semanas - 1) * ((2 * u_max) ** 2))
        return ((w1 * C_crit) + (w2 * C_dev) + (w3 * C_smooth),)

    toolbox.register("evaluate", evaluar_srs)
    toolbox.register("mate", tools.cxTwoPoint)

    def mutar_nivel(individuo, indpb):
        resultado = []
        for valor in individuo:
            if np.random.random() < indpb:
                resultado.append(np.random.choice(niveles))
            else:
                resultado.append(valor)
        return creator.Individual(resultado),

    toolbox.register("mutate", mutar_nivel, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=4)

    pop = toolbox.population(n=200)
    individuo_base = creator.Individual([0.0] * semanas)
    pop[0] = individuo_base
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=80, stats=stats, halloffame=hof, verbose=False)

    mejor_secuencia = np.array(hof[0])
    base_lineal = np.zeros(semanas)
    mejora_pct = max(0.0, 100.0 * (1 - np.mean(np.abs(mejor_secuencia - base_lineal)) / max(np.mean(np.abs(base_lineal)), 1e-6)))

    fechas = pd.date_range(start=pd.Timestamp(fecha_inicio), periods=semanas, freq="W-SUN")
    datos_opt = pd.DataFrame({"Fecha": fechas, "R_obs": R_obs, "Delta_S_obs": Delta_S_obs, "u_opt": mejor_secuencia})
    flujo = datos_opt["Delta_S_obs"].to_numpy() - datos_opt["u_opt"].to_numpy()
    s_acumulado = np.zeros(semanas + 1)
    s_acumulado[0] = S_inicial
    for idx, valor in enumerate(flujo):
        s_acumulado[idx + 1] = s_acumulado[idx] + valor
    datos_opt["S_simulada"] = s_acumulado[1:]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(datos_opt["Fecha"], datos_opt["R_obs"], label="R original", color="tab:blue")
    axes[0].plot(datos_opt["Fecha"], datos_opt["u_opt"], label="Optimización genética", color="tab:orange")
    axes[0].set_title("Comparación original vs optimización")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(datos_opt["Fecha"], datos_opt["S_simulada"], label="Storage simulado", color="tab:green")
    axes[1].axhline(S_min, color="red", linestyle="--", label="Nivel crítico")
    axes[1].set_title("Mejora genética")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    ruta_grafica = base_dir / "resultado_optimizacion.png"
    fig.savefig(ruta_grafica, dpi=160)
    plt.close(fig)

    return {
        "fecha_inicio": pd.Timestamp(fecha_inicio),
        "semanas": semanas,
        "mejor_secuencia": mejor_secuencia,
        "mejora_genetica": {
            "mejora_pct": round(float(mejora_pct), 2),
            "score": round(float(hof[0].fitness.values[0]), 6),
            "ruta_grafica": str(ruta_grafica),
        },
        "ruta_grafica": str(ruta_grafica),
    }


def ejecutar_interfaz() -> None:
    import tkinter as tk
    from tkinter import simpledialog, messagebox

    base_dir = Path(__file__).resolve().parent
    fechas_disponibles = obtener_fechas_disponibles(base_dir)
    if not fechas_disponibles:
        messagebox.showerror("Sin datos", "No se encontraron fechas válidas en los datasets.")
        return

    root = tk.Tk()
    root.withdraw()

    fecha_str = simpledialog.askstring(
        "Ventana histórica",
        "Selecciona la fecha inicial de la ventana histórica:",
        initialvalue=str(fechas_disponibles[0].date()),
    )
    if not fecha_str:
        return

    try:
        fecha_inicio = pd.Timestamp(fecha_str)
    except Exception:
        messagebox.showerror("Fecha inválida", "Ingresa una fecha válida en formato YYYY-MM-DD.")
        return

    opciones = [7, 26, 52]
    semanas = simpledialog.askinteger(
        "Semanas",
        "¿Cuántas semanas tomar en cuenta? (7, 26 o 52)",
        initialvalue=26,
        minvalue=7,
        maxvalue=52,
    )
    if semanas is None:
        return

    if semanas not in opciones:
        semanas = 26 if semanas < 26 else 52

    resultado = construir_resultado_ventana(base_dir, fecha_inicio.strftime("%Y-%m-%d"), semanas)

    messagebox.showinfo(
        "Resultados",
        f"Ventana: {resultado['fecha_inicio'].date()}\nSemanas: {resultado['semanas']}\nMejora estimada: {resultado['mejora_genetica']['mejora_pct']}%\nGráfica generada en: {resultado['ruta_grafica']}",
    )

    if os.path.exists(resultado["ruta_grafica"]):
        from PIL import Image
        img = Image.open(resultado["ruta_grafica"])
        img.show()


if __name__ == "__main__":
    ejecutar_interfaz()
