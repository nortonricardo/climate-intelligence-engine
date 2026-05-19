"""
1.4 — Limpeza e separação das variáveis meteorológicas.

Para cada variável-alvo é gerado um parquet independente com a estrutura:

    code        — código da estação (ex: A565)
    time        — data/hora da medição (UTC)
    measurement — valor observado

Variáveis geradas:
    data/temperature.parquet
    data/humidity.parquet
    data/rainfall.parquet
    data/global_radiation.parquet
    data/pressure.parquet

Regras de limpeza aplicadas:
    1. Registros com measurement = NaN são removidos.
    2. global_radiation: valores negativos são substituídos por 0.
       Radiação negativa é impossível fisicamente — representa ausência
       de luz (noite / sensor em repouso) e deve ser tratada como zero.

Input:
    data/weather_measurements.parquet

Usage:
    python 1.4_clean_data.py
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR   = Path(__file__).parent / "data"
INPUT_PATH = DATA_DIR / "weather_measurements.parquet"

VARIABLES = [
    "temperature",
    "humidity",
    "rainfall",
    "global_radiation",
    "pressure",
]


def load_measurements() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run 1.1_download_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(
        INPUT_PATH,
        columns=["station_code", "measurement_time"] + VARIABLES,
    )
    df["measurement_time"] = pd.to_datetime(df["measurement_time"])
    print(f"Registros carregados : {len(df):,}")
    return df


def build_variable_df(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    out = (
        df[["station_code", "measurement_time", variable]]
        .rename(columns={
            "station_code":     "code",
            "measurement_time": "time",
            variable:           "measurement",
        })
        .dropna(subset=["measurement"])
        .reset_index(drop=True)
    )

    if variable == "global_radiation":
        negative = (out["measurement"] < 0).sum()
        if negative:
            out["measurement"] = out["measurement"].clip(lower=0)
            print(f"  global_radiation: {negative:,} valores negativos → 0")

    return out


def save(df: pd.DataFrame, variable: str) -> None:
    path = DATA_DIR / f"{variable}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    size_mb = path.stat().st_size / 1_048_576
    print(f"  {variable:<20} {len(df):>12,} registros   {size_mb:.1f} MB")


def main() -> None:
    print("=== 1.4 Clean Data ===\n")
    measurements = load_measurements()

    print("\nProcessando variáveis:")
    for variable in VARIABLES:
        clean = build_variable_df(measurements, variable)
        save(clean, variable)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
