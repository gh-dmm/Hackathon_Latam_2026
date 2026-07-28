from pathlib import Path

import numpy as np

from hackatonlatam2026.data_utils import cargar_datos_hidrologicos
from hackatonlatam2026.genetico_v3 import preparar_ventana_semanal


def test_carga_de_datos_hidrologicos():
    base_dir = Path(__file__).resolve().parents[1] / "hackatonlatam2026"
    datos = cargar_datos_hidrologicos(base_dir)

    assert set(datos) >= {"lib", "cambio", "total", "evap", "batimetria"}
    assert not datos["lib"].empty
    assert not datos["cambio"].empty
    assert not datos["total"].empty
    assert not datos["batimetria"].empty


def test_preparar_ventana_semanal_devuelve_arrays_coherentes():
    base_dir = Path(__file__).resolve().parents[1] / "hackatonlatam2026"
    datos = cargar_datos_hidrologicos(base_dir)

    R_obs, Delta_S_obs, S_inicial = preparar_ventana_semanal(
        datos["lib"],
        datos["cambio"],
        datos["total"],
        datos["evap"],
        datos["batimetria"],
        "2024-01-01",
        27,
    )

    assert len(R_obs) == 27
    assert len(Delta_S_obs) == 27
    assert np.isfinite(S_inicial)
    assert np.isfinite(R_obs).all()
    assert np.isfinite(Delta_S_obs).all()
