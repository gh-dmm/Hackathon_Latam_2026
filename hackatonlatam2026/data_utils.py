from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


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

    return df


def cargar_datos_hidrologicos(base_dir=None):
    base_dir = Path(base_dir or BASE_DIR)

    def read_csv(path):
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path)
        return pd.read_csv(path, engine="python", encoding="utf-8-sig")

    df_lib = read_csv(base_dir / "R_observ.xlsx")
    df_cambio = read_csv(base_dir / "Cambio_almacenamiento_historico.csv")
    df_total = read_csv(base_dir / "DataSetExport-Total Storage.csv")
    df_evap = read_csv(base_dir / "DataSetExport-Evaporation,accumltd.Daily Evaporation - mm@08461200-Instantaneous-mm-20260622185804.csv")
    df_batimetria = read_csv(base_dir / "tabla_elevacion_volumen_FINAL.csv")

    return {
        "lib": limpiar_dataset(df_lib),
        "cambio": limpiar_dataset(df_cambio),
        "total": limpiar_dataset(df_total),
        "evap": limpiar_dataset(df_evap),
        "batimetria": limpiar_dataset(df_batimetria),
    }
