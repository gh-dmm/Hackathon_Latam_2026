from pathlib import Path

import pandas as pd

from hackatonlatam2026.gui_app import (
    construir_pagina_web,
    construir_resultado_ventana,
    obtener_fechas_disponibles,
    procesar_analisis_web,
)


def test_obtener_fechas_disponibles_devuelve_valores_reales():
    base_dir = Path(__file__).resolve().parents[1] / "hackatonlatam2026"
    fechas = obtener_fechas_disponibles(base_dir)

    assert len(fechas) > 0
    assert all(isinstance(fecha, pd.Timestamp) for fecha in fechas)


def test_construir_resultado_ventana_devuelve_metricas_coherentes():
    base_dir = Path(__file__).resolve().parents[1] / "hackatonlatam2026"
    resultado = construir_resultado_ventana(base_dir, "2024-01-01", 7)

    assert resultado["semanas"] == 7
    assert resultado["fecha_inicio"] == pd.Timestamp("2024-01-01")
    assert resultado["mejor_secuencia"].shape[0] == 7
    assert resultado["mejora_genetica"]["mejora_pct"] >= 0


def test_construir_pagina_web_incluye_formulario_y_opciones():
    base_dir = Path(__file__).resolve().parents[1] / "hackatonlatam2026"
    fechas = obtener_fechas_disponibles(base_dir)
    html = construir_pagina_web(fechas, base_dir)

    assert "id=\"analisis-form\"" in html
    assert "7 semanas" in html
    assert "52 semanas" in html


def test_procesar_analisis_web_devuelve_datos_de_resultado():
    base_dir = Path(__file__).resolve().parents[1] / "hackatonlatam2026"
    resultado = procesar_analisis_web({"fecha": "2024-01-01", "semanas": "7"}, base_dir)

    assert resultado["semanas"] == 7
    assert resultado["mejora_genetica"]["mejora_pct"] >= 0
