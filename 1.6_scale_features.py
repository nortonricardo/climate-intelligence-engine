"""
1.6 — Split treino/teste e normalização MinMax (0–1).

Para cada variável:
  1. Filtra dados a partir de START_YEAR (2002 = primeiro ano com ≥15 vizinhos).
  2. Remove colunas dos slots 16-20 (K fixo em 15).
  3. Remove linhas com qualquer NaN em n01..n15 (após drop — mais seguro).
  4. Ajusta MinMaxScaler com clip de percentil (CLIP_PCT) para robustez a outliers.
     x_scaled = clip((x − p1) / (p99 − p1), 0, 1)
  5. Aplica scaler no dataset completo — exceto sin/cos (já em [−1,1]), code, time.
  6. Divide temporalmente pelo TRAIN_RATIO (80/20) e ordena cada split por tempo.

Sin/cos não são escalados: hour_sin/cos, doy_sin/cos e b{i}_sin/cos são
matematicamente limitados a [−1, 1] e não precisam de normalização.

Input:
    data/{variable}_neighbors.parquet

Output:
    data/{variable}_train_scaled.parquet
    data/{variable}_test_scaled.parquet
    models/1.6_scaler_{variable}.scaler

Usage:
    python 1.6_scale_features.py
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import DATA_DIR, MODELS_DIR, VARIABLES

K           = 15     # vizinhos usados; slots 16-20 são descartados
START_YEAR  = 2002   # primeiro ano com ≥ 15 vizinhos disponíveis
CLIP_PCT    = 1.0    # percentil de corte para outliers (1% inf e 1% sup)
TRAIN_RATIO = 0.8

VARIABLES_TO_RUN = None  # None = todas; ou ex: ["temperature"]

# ── colunas derivadas de K ────────────────────────────────────────────────────

_K_MAX = 20
DROP_COLS = (
    [f"n{i+1:02d}"     for i in range(K, _K_MAX)]
    + [f"d{i+1:02d}"   for i in range(K, _K_MAX)]
    + [f"a{i+1:02d}"   for i in range(K, _K_MAX)]
    + [f"b{i+1:02d}_sin" for i in range(K, _K_MAX)]
    + [f"b{i+1:02d}_cos" for i in range(K, _K_MAX)]
)

NEIGHBOR_COLS = [f"n{i+1:02d}" for i in range(K)]  # colunas para filtro NaN

# sin/cos já em [−1, 1] — não precisam de scaler
SIN_COS_COLS = set(
    [f"b{i+1:02d}_sin" for i in range(K)]
    + [f"b{i+1:02d}_cos" for i in range(K)]
    + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
)
SKIP_COLS = {"code", "time"} | SIN_COS_COLS


# ── scaler MinMax com clip de percentil ───────────────────────────────────────

def fit_scaler(df: pd.DataFrame, cols: list[str]) -> MinMaxScaler:
    """
    Cria um MinMaxScaler robusto a outliers:
    clippa os dados ao intervalo [p1, p99] antes de ajustar,
    para que o scaler não seja dominado por valores extremos.
    """
    data   = df[cols].to_numpy(dtype=np.float64)
    lowers = np.nanpercentile(data, CLIP_PCT,       axis=0)
    uppers = np.nanpercentile(data, 100 - CLIP_PCT, axis=0)
    data_clipped = np.clip(data, lowers, uppers)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_clipped)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, cols: list[str]) -> pd.DataFrame:
    """Transforma as colunas com o scaler ajustado. Modifica df in-place."""
    data = df[cols].to_numpy(dtype=np.float64)
    data = scaler.transform(data)        # já clippa para [0,1] pelo fit
    np.clip(data, 0.0, 1.0, out=data)   # garante [0,1] mesmo para outliers reais
    df[cols] = data.astype(np.float32)
    return df


# ── split temporal ────────────────────────────────────────────────────────────

def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    times      = np.sort(df["time"].unique())
    cutoff_idx = max(1, int(len(times) * TRAIN_RATIO)) - 1
    cutoff     = times[cutoff_idx]
    train_df   = df[df["time"] <= cutoff].sort_values("time").reset_index(drop=True)
    test_df    = df[df["time"] >  cutoff].sort_values("time").reset_index(drop=True)
    return train_df, test_df, str(pd.Timestamp(cutoff).date())


# ── processamento por variável ────────────────────────────────────────────────

def process_variable(variable: str) -> None:
    path = DATA_DIR / f"{variable}_neighbors.parquet"
    if not path.exists():
        print(f"  SKIP {variable}: {path.name} não encontrado — rode 1.5 primeiro.")
        return

    def _step(msg: str) -> None:
        print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)

    t0 = time.time()
    print(f"\n[{variable}]", flush=True)

    # 1. lê parquet
    _step("lendo parquet...")
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    _step(f"parquet lido — {len(df):,} linhas × {df.shape[1]} colunas")

    # 2. filtro de ano
    n_raw = len(df)
    df = df[df["time"].dt.year >= START_YEAR].reset_index(drop=True)
    _step(f"ano >= {START_YEAR}: {n_raw:,} → {len(df):,} registros")

    # 3. drop slots 16-20
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    _step(f"colunas após drop slots 16-20: {df.shape[1]}")

    # 4. remove linhas com NaN em n01..n15 (após drop — confirma que k=15 está intacto)
    n_before = len(df)
    df = df.dropna(subset=NEIGHBOR_COLS).reset_index(drop=True)
    _step(f"dropna n01..n15: {n_before:,} → {len(df):,} ({n_before - len(df):,} removidos)")

    if df.empty:
        print(f"  SKIP {variable}: nenhum registro após filtros.")
        return

    # 5. fit scaler no dataset completo
    _step(f"ajustando MinMax scaler (clip={CLIP_PCT}%)...")
    cols   = [c for c in df.columns if c not in SKIP_COLS]
    scaler = fit_scaler(df, cols)

    MODELS_DIR.mkdir(exist_ok=True)
    scaler_path = MODELS_DIR / f"scaler_{variable}.scaler"
    joblib.dump({"scaler": scaler, "cols": cols, "clip_pct": CLIP_PCT}, scaler_path)
    _step(f"scaler salvo → {scaler_path.name}  ({len(cols)} colunas)")

    # 6. aplica scaler no df todo
    _step(f"aplicando scaler ({len(df):,} linhas × {len(cols)} colunas)...")
    df = apply_scaler(df, scaler, cols)

    # 7. split temporal (ordena dentro do split_by_time)
    _step("split treino/teste...")
    train_df, test_df, cutoff = split_by_time(df)
    del df
    _step(f"treino até {cutoff} | {len(train_df):,} treino | {len(test_df):,} teste")

    # 8. salva
    for split_df, split_name in [(train_df, "train"), (test_df, "test")]:
        out_path = DATA_DIR / f"{variable}_{split_name}_scaled.parquet"
        _step(f"gravando {out_path.name}...")
        split_df.to_parquet(out_path, index=False, compression="snappy", engine="pyarrow")
        size_mb = out_path.stat().st_size / 1_048_576
        _step(f"→ {out_path.name}   {size_mb:.0f} MB")
        del split_df

    _step(f"CONCLUÍDO em {time.time() - t0:.0f}s total")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"=== 1.6 MinMax Scale + Split (train={TRAIN_RATIO:.0%} / test={1-TRAIN_RATIO:.0%}) ===")
    print(f"  clip_pct={CLIP_PCT}%  |  start_year={START_YEAR}  |  K={K} vizinhos")
    print(f"  sin/cos não escalados: {len(SIN_COS_COLS)} colunas\n")

    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    for variable in variables:
        process_variable(variable)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
