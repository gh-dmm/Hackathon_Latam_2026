from __future__ import annotations

import html
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

try:
    from .data_utils import cargar_datos_hidrologicos
    from .genetico_v3 import preparar_ventana_semanal
except ImportError:  # pragma: no cover - fallback para ejecución directa
    from data_utils import cargar_datos_hidrologicos
    from genetico_v3 import preparar_ventana_semanal


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
    datos_opt["R_opt"] = datos_opt["R_obs"] + datos_opt["u_opt"]
    flujo = datos_opt["Delta_S_obs"].to_numpy() - datos_opt["u_opt"].to_numpy()
    s_acumulado = np.zeros(semanas + 1)
    s_acumulado[0] = S_inicial
    for idx, valor in enumerate(flujo):
        s_acumulado[idx + 1] = s_acumulado[idx] + valor
    datos_opt["S_simulada"] = s_acumulado[1:]

    ajustes_abs = np.abs(mejor_secuencia)
    ajuste_total = float(np.sum(ajustes_abs))
    semanas_aumento = int(np.sum(mejor_secuencia > 0))
    semanas_disminucion = int(np.sum(mejor_secuencia < 0))
    semanas_sin_cambio = int(np.sum(mejor_secuencia == 0))
    max_ajuste = float(np.max(mejor_secuencia))
    min_ajuste = float(np.min(mejor_secuencia))
    descripcion_ajustes = (
        "El optimizador ajusta la liberación original para generar la serie optimizada "
        "R_opt = R_obs + u_opt."
    )
    if ajuste_total == 0.0:
        descripcion_ajustes = (
            "El optimizador no aplicó ningún ajuste: la serie optimizada coincide con la original. "
            "Esto puede suceder cuando cualquier desviación viola una restricción o no mejora la función objetivo."
        )
    else:
        descripcion_ajustes = (
            f"El optimizador realizó ajustes en {semanas - semanas_sin_cambio} semanas, "
            f"con {semanas_aumento} semanas de incremento y {semanas_disminucion} semanas de reducción. "
            f"El mayor ajuste fue {max_ajuste:.2f} y el menor ajuste fue {min_ajuste:.2f}."
        )

    fig_comp, ax_comp = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax_comp.plot(datos_opt["Fecha"], datos_opt["R_obs"], label="R original", color="tab:blue")
    ax_comp.plot(datos_opt["Fecha"], datos_opt["R_opt"], label="R optimizado", color="tab:orange")
    ax_comp.set_title("Comparación original vs optimización")
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)

    fig_mejora, ax_mejora = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax_mejora.plot(datos_opt["Fecha"], datos_opt["S_simulada"], label="Storage simulado", color="tab:green")
    ax_mejora.axhline(S_min, color="red", linestyle="--", label="Nivel crítico")
    ax_mejora.set_title("Mejora genética")
    ax_mejora.legend()
    ax_mejora.grid(True, alpha=0.3)

    ruta_comparacion = base_dir / f"comparacion_{pd.Timestamp(fecha_inicio).strftime('%Y%m%d')}_{semanas}_semanas.png"
    ruta_mejora = base_dir / f"mejora_{pd.Timestamp(fecha_inicio).strftime('%Y%m%d')}_{semanas}_semanas.png"
    fig_comp.savefig(ruta_comparacion, dpi=160)
    fig_mejora.savefig(ruta_mejora, dpi=160)
    plt.close(fig_comp)
    plt.close(fig_mejora)

    return {
        "fecha_inicio": pd.Timestamp(fecha_inicio),
        "semanas": semanas,
        "mejor_secuencia": mejor_secuencia,
        "movimientos": {
            "ajuste_total": round(ajuste_total, 2),
            "max_ajuste": max_ajuste,
            "min_ajuste": min_ajuste,
            "semanas_aumento": semanas_aumento,
            "semanas_disminucion": semanas_disminucion,
            "semanas_sin_cambio": semanas_sin_cambio,
            "descripcion": descripcion_ajustes,
        },
        "mejora_genetica": {
            "mejora_pct": round(float(mejora_pct), 2),
            "score": round(float(hof[0].fitness.values[0]), 6),
            "ruta_grafica": str(ruta_comparacion),
        },
        "ruta_grafica": str(ruta_comparacion),
        "rutas_graficas": {
            "comparacion": str(ruta_comparacion),
            "mejora": str(ruta_mejora),
        },
    }


def construir_pagina_web(fechas_disponibles: List[pd.Timestamp] | List[str], base_dir: Path | None = None) -> str:
    fechas = []
    for fecha in fechas_disponibles:
        if isinstance(fecha, pd.Timestamp):
            fechas.append(fecha.to_pydatetime().date().isoformat())
        else:
            fechas.append(str(fecha).split(" ")[0])

    if not fechas:
        fechas = ["2024-01-01"]

    opciones = "".join(
        f'<option value="{fecha}"{" selected" if idx == 0 else ""}>{fecha}</option>'
        for idx, fecha in enumerate(fechas)
    )
    opciones_semanas = "".join(
        f'<option value="{valor}"{" selected" if valor == 26 else ""}>{valor} semanas</option>'
        for valor in (7, 26, 52)
    )

    return f"""<!DOCTYPE html>
<html lang=\"es\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Optimización genética local</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f7fb; color: #1f2937; margin: 0; padding: 2rem; }}
    .card {{ max-width: 760px; margin: 0 auto; background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
    h1 {{ margin-top: 0; }}
    form {{ display: grid; gap: 1rem; }}
    label {{ font-weight: 600; }}
    select, button {{ padding: 0.7rem 0.95rem; font-size: 1rem; border-radius: 8px; border: 1px solid #cbd5e1; }}
    button {{ background: #2563eb; color: white; border: none; cursor: pointer; }}
    .hint {{ color: #64748b; font-size: 0.95rem; }}
    .result {{ margin-top: 1.5rem; padding: 1rem; border-radius: 12px; background: #eff6ff; }}
    img {{ width: 100%; height: auto; margin-top: 1rem; border-radius: 12px; border: 1px solid #dbeafe; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Optimización genética local</h1>
    <p class=\"hint\">Selecciona una fecha histórica y el número de semanas para ejecutar el análisis y generar las gráficas.</p>
    <form id=\"analisis-form\" action=\"/analizar\" method=\"post\">
      <label for=\"fecha\">Fecha inicial</label>
      <select id=\"fecha\" name=\"fecha\">
        {opciones}
      </select>
      <label for=\"semanas\">Semanas a analizar</label>
      <select id=\"semanas\" name=\"semanas\">
        {opciones_semanas}
      </select>
      <button type=\"submit\">Generar análisis</button>
    </form>
    <div class=\"result\">La interfaz se ejecuta de forma local en el puerto 8000 y genera archivos PNG en la carpeta del proyecto.</div>
  </div>
</body>
</html>"""


def construir_pagina_resultado(resultado: Dict[str, Any], base_dir: Path | None = None) -> str:
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    comparacion = Path(resultado["rutas_graficas"]["comparacion"]).name
    mejora = Path(resultado["rutas_graficas"]["mejora"]).name
    return f"""<!DOCTYPE html>
<html lang=\"es\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Resultados del análisis</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f7fb; color: #1f2937; margin: 0; padding: 2rem; }}
    .card {{ max-width: 960px; margin: 0 auto; background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
    h1 {{ margin-top: 0; }}
    .metric {{ margin-bottom: 0.8rem; }}
    .section {{ margin-top: 1.5rem; }}
    .summary {{ background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 12px; padding: 1rem; margin-top: 1rem; }}
    .summary p {{ margin: 0.4rem 0; }}
    img {{ width: 100%; height: auto; margin: 1rem 0; border-radius: 12px; border: 1px solid #dbeafe; }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Resultados del análisis</h1>
    <div class=\"metric\"><strong>Fecha:</strong> {resultado['fecha_inicio'].date()}</div>
    <div class=\"metric\"><strong>Semanas:</strong> {resultado['semanas']}</div>
    <div class=\"metric\"><strong>Mejora estimada:</strong> {resultado['mejora_genetica']['mejora_pct']}%</div>
    <div class=\"metric\"><strong>Score:</strong> {resultado['mejora_genetica']['score']}</div>
    <p><a href=\"/\">Volver al formulario</a></p>

    <div class=\"section\">
      <h2>Descripción de los movimientos de optimización</h2>
      <div class=\"summary\">
        <p>{html.escape(resultado['movimientos']['descripcion'])}</p>
        <p><strong>Ajuste total absoluto:</strong> {resultado['movimientos']['ajuste_total']:.2f}</p>
        <p><strong>Máximo ajuste:</strong> {resultado['movimientos']['max_ajuste']:.2f}</p>
        <p><strong>Mínimo ajuste:</strong> {resultado['movimientos']['min_ajuste']:.2f}</p>
        <p><strong>Semanas de aumento:</strong> {resultado['movimientos']['semanas_aumento']}</p>
        <p><strong>Semanas de disminución:</strong> {resultado['movimientos']['semanas_disminucion']}</p>
        <p><strong>Semanas sin cambio:</strong> {resultado['movimientos']['semanas_sin_cambio']}</p>
      </div>
    </div>

    <div class=\"section\">
      <h2>Comparación original vs optimización</h2>
      <img src=\"/{comparacion}\" alt=\"Gráfica comparación\" />
    </div>

    <div class=\"section\">
      <h2>Mejora genética</h2>
      <img src=\"/{mejora}\" alt=\"Gráfica mejora genética\" />
    </div>
  </div>
</body>
</html>"""


def procesar_analisis_web(payload: Dict[str, Any] | None, base_dir: Path | None = None) -> Dict[str, Any]:
    datos = payload or {}
    fecha_str = datos.get("fecha") or datos.get("fecha_inicio") or "2024-01-01"
    semanas = datos.get("semanas", "26")
    try:
        semanas_int = int(semanas)
    except (TypeError, ValueError):
        semanas_int = 26
    return construir_resultado_ventana(base_dir, fecha_str, semanas_int)


class _WebAppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(construir_pagina_web(obtener_fechas_disponibles(self.base_dir)))
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        self._serve_file(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/analizar":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        payload = {key: values[0] if values else "" for key, values in parse_qs(body, keep_blank_values=True).items()}
        resultado = procesar_analisis_web(payload, self.base_dir)
        self._send_html(construir_pagina_resultado(resultado, self.base_dir))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_html(self, html_text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_text.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html_text.encode("utf-8"))

    def _send_json(self, payload: Dict[str, Any]) -> None:
        body = str(payload).replace("'", '"')
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _serve_file(self, path: str) -> None:
        if path.startswith("/"):
            path = path[1:]
        file_path = (self.base_dir / path).resolve()
        if not str(file_path).startswith(str(self.base_dir.resolve())):
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", self._content_type(file_path))
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with file_path.open("rb") as handle:
            self.wfile.write(handle.read())

    @staticmethod
    def _content_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix == ".jpg" or suffix == ".jpeg":
            return "image/jpeg"
        if suffix == ".css":
            return "text/css; charset=utf-8"
        return "application/octet-stream"


def _build_handler(base_dir: Path) -> type[_WebAppHandler]:
    class Handler(_WebAppHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.base_dir = base_dir
            super().__init__(*args, **kwargs)

    return Handler


def ejecutar_interfaz(host: str = "127.0.0.1", port: int = 8000) -> None:
    base_dir = Path(__file__).resolve().parent
    handler_cls = _build_handler(base_dir)
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"Interfaz web local lista en http://{host}:{port}")
    print("Abre esa URL en tu navegador para usar la aplicación.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Servidor detenido.")


if __name__ == "__main__":
    ejecutar_interfaz()
