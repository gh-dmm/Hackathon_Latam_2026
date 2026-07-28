from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def _leer_csv_robusto(path):
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    with open(path, "r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()

    header_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if not stripped:
            continue
        if any(token in stripped for token in ["timestamp", "fecha", "inicio de intervalo", "fin del intervalo", "valor", "value", "elevation_m", "volume_tcm"]):
            header_idx = idx
            break

    if header_idx is None:
        return pd.read_csv(path, engine="python", encoding="utf-8-sig", low_memory=False)

    sample = "".join(lines[header_idx:header_idx + 3])
    sep = ";" if ";" in sample else ","
    return pd.read_csv(path, sep=sep, skiprows=header_idx, engine="python", encoding="utf-8-sig")


def limpiar_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Timestamp (UTC-06:00)" in df.columns:
        df.rename(columns={"Timestamp (UTC-06:00)": "Fecha"}, inplace=True)
    if "Inicio de intervalo (UTC-06:00)" in df.columns:
        df.rename(columns={"Inicio de intervalo (UTC-06:00)": "Fecha"}, inplace=True)
    if "Fin del intervalo (UTC-06:00)" in df.columns:
        df.rename(columns={"Fin del intervalo (UTC-06:00)": "Fecha"}, inplace=True)

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Fecha"]).set_index("Fecha")
    elif "Timestamp (UTC-06:00)" in df.columns:
        df.rename(columns={"Timestamp (UTC-06:00)": "Fecha"}, inplace=True)
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Fecha"]).set_index("Fecha")

    if "Value (mm)" in df.columns:
        df.rename(columns={"Value (mm)": "Evaporacion_mm"}, inplace=True)
    elif "Value (m)" in df.columns:
        df.rename(columns={"Value (m)": "Valor"}, inplace=True)

    if "elevation_m" not in df.columns and "elevation" in df.columns:
        df.rename(columns={"elevation": "elevation_m"}, inplace=True)
    if "volume_TCM" not in df.columns and "volume" in df.columns:
        df.rename(columns={"volume": "volume_TCM"}, inplace=True)

    return df


def cargar_datos_hidrologicos(base_dir=None):
    base_dir = Path(base_dir or BASE_DIR)

    df_lib = _leer_csv_robusto(base_dir / "R_observ.xlsx")
    df_cambio = _leer_csv_robusto(base_dir / "Cambio_almacenamiento_historico.csv")
    df_total = _leer_csv_robusto(base_dir / "DataSetExport-Total Storage.csv")
    df_evap = _leer_csv_robusto(base_dir / "DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200-Instantaneous-mm-20260622185804.csv")
    df_batimetria = _leer_csv_robusto(base_dir / "tabla_elevacion_volumen_FINAL.csv")

    return {
        "lib": limpiar_dataset(df_lib),
        "cambio": limpiar_dataset(df_cambio),
        "total": limpiar_dataset(df_total),
        "evap": limpiar_dataset(df_evap),
        "batimetria": limpiar_dataset(df_batimetria),
    }
