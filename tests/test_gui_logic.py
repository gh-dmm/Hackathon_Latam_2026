from pathlib import Path

import pandas as pd

from hackatonlatam2026.gui_app import construir_resultado_ventana, obtener_fechas_disponibles


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
