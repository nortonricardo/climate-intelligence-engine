"""
1.6 — Split treino/teste e normalização (StandardScaler).

Para cada variável:
  1. Filtra dados a partir de START_YEAR — período com rede densa o suficiente.
  2. Remove linhas onde menos de MIN_STATIONS vizinhos têm dado disponível
     (n01..n{MIN_STATIONS} NaN). Após esse filtro, n01..n15 são garantidamente
     não-NaN em todas as linhas do dataset resultante.
  3. Divide temporalmente o conjunto usável pelo 80º percentil de timestamps:
       treino — timestamps até o corte  (TRAIN_RATIO = 0.8)
       teste  — timestamps após o corte (0.2)
     O corte é feito por timestamp único, garantindo que nenhuma estação
     apareça no mesmo período em treino e teste.
  4. Ajusta o StandardScaler SOMENTE no conjunto de treino (evita leakage).
  5. Transforma treino e teste com o mesmo scaler.

Scaler salvo como JSON — inclui metadados do split e estatísticas por coluna.
Para inverse transform: x_orig = x_scaled * std + mean.

ACELERAÇÃO GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transformação aplicada em chunks de CHUNK_SIZE linhas na GPU (float32).
Com 5 GPUs e 5 variáveis, cada variável ocupa uma GPU via multiprocessing.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
    data/{variable}_neighbors.parquet

Output:
    data/{variable}_train_scaled.parquet
    data/{variable}_test_scaled.parquet
    models/1.6_scaler_{variable}.json

Usage:
    python 1.6_scale_features.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

N_GPUS       = 5
CHUNK_SIZE   = 5_000_000  # linhas por chunk GPU (~2 GB em float32)
MIN_STATIONS = 15          # vizinhos garantidos por linha; slots 16-20 são descartados
K_NEIGHBORS  = 20          # total de slots no arquivo _neighbors (gerado pelo 1.5)
START_YEAR   = 2010        # descarta dados antes deste ano (rede ainda esparsa)
TRAIN_RATIO  = 0.8         # 80% treino, 20% teste (split temporal)

# Colunas dos slots 16-20 que serão removidas antes de salvar
DROP_COLS = (
    [f"n{i+1:02d}"     for i in range(MIN_STATIONS, K_NEIGHBORS)]
    + [f"d{i+1:02d}"   for i in range(MIN_STATIONS, K_NEIGHBORS)]
    + [f"a{i+1:02d}"   for i in range(MIN_STATIONS, K_NEIGHBORS)]
    + [f"b{i+1:02d}_sin" for i in range(MIN_STATIONS, K_NEIGHBORS)]
    + [f"b{i+1:02d}_cos" for i in range(MIN_STATIONS, K_NEIGHBORS)]
)

VARIABLES_TO_RUN = None  # None = todas; ou ex: ["temperature"]

VARIABLES = [
    "temperature",
    "humidity",
    "rainfall",
    "global_radiation",
    "pressure",
]

SKIP_COLS = {"code", "time"}


# ── filtros e split temporal ──────────────────────────────────────────────────

def filter_usable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas antes de START_YEAR e linhas com menos de MIN_STATIONS
    vizinhos disponíveis. Após esse filtro, n01..n{MIN_STATIONS} são
    garantidamente não-NaN em todo o dataset.
    """
    df = df[df["time"].dt.year >= START_YEAR]
    threshold_cols = [f"n{i+1:02d}" for i in range(MIN_STATIONS)]
    mask = df[threshold_cols].notna().sum(axis=1) >= MIN_STATIONS
    return df[mask].reset_index(drop=True)


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Divide df pelo 80º percentil dos timestamps únicos.
    Retorna (train_df, test_df, cutoff_date_str).
    """
    times      = np.sort(df["time"].unique())
    cutoff_idx = max(1, int(len(times) * TRAIN_RATIO)) - 1
    cutoff     = times[cutoff_idx]
    train_df   = df[df["time"] <= cutoff]
    test_df    = df[df["time"] >  cutoff]
    return train_df, test_df, str(pd.Timestamp(cutoff).date())


# ── scaler manual ─────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Computa mean e std por coluna ignorando NaN.
    Deve ser chamado SOMENTE com dados de treino.
    """
    stats: dict[str, dict] = {}
    for col in cols:
        vals = df[col].to_numpy(dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            stats[col] = {"mean": 0.0, "std": 1.0}
            continue
        mean = float(vals.mean())
        std  = float(vals.std())
        stats[col] = {"mean": mean, "std": std if std > 0 else 1.0}
    return stats


def apply_scaler(
    df: pd.DataFrame,
    stats: dict,
    gpu_id: int = 0,
) -> pd.DataFrame:
    """
    Aplica (x − mean) / std em todas as colunas do stats.
    NaN permanece NaN. Opera em float32. GPU com chunks.
    """
    cols  = list(stats.keys())
    means = np.array([stats[c]["mean"] for c in cols], dtype=np.float32)
    stds  = np.array([stats[c]["std"]  for c in cols], dtype=np.float32)
    data  = df[cols].values.astype(np.float32)

    if HAS_CUPY:
        cp.cuda.Device(gpu_id).use()
        means_gpu = cp.asarray(means)
        stds_gpu  = cp.asarray(stds)
        for s in range(0, len(data), CHUNK_SIZE):
            chunk = cp.asarray(data[s : s + CHUNK_SIZE])
            data[s : s + CHUNK_SIZE] = cp.asnumpy((chunk - means_gpu) / stds_gpu)
    else:
        for s in range(0, len(data), CHUNK_SIZE):
            data[s : s + CHUNK_SIZE] = (data[s : s + CHUNK_SIZE] - means) / stds

    out       = df.copy()
    out[cols] = data
    return out


# ── processamento por variável ────────────────────────────────────────────────

def process_variable(variable: str, gpu_id: int = 0) -> None:
    path = DATA_DIR / f"{variable}_neighbors.parquet"
    if not path.exists():
        print(f"  SKIP {variable}: {path.name} não encontrado — rode 1.5 primeiro.")
        return

    if HAS_CUPY:
        cp.cuda.Device(gpu_id).use()

    t0    = time.time()
    label = f"[GPU {gpu_id}][{variable}]" if HAS_CUPY else f"[{variable}]"
    print(f"\n{label}")

    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 1. filtros: ano + mínimo de vizinhos
    n_raw = len(df)
    df    = filter_usable(df)
    if df.empty:
        print(f"  SKIP {variable}: nenhum registro após filtros (START_YEAR={START_YEAR}, MIN_STATIONS={MIN_STATIONS}).")
        return
    data_start = str(df["time"].min().date())
    print(f"  {n_raw:,} → {len(df):,} registros  (a partir de {data_start}, ≥{MIN_STATIONS} vizinhos)")

    # descarta slots 16-20 (além do mínimo garantido)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"  Colunas após drop slots 16-20: {df.shape[1]}")

    # 2. split temporal
    train_df, test_df, cutoff = split_by_time(df)
    print(f"  train até {cutoff}  |  {len(train_df):,} treino  |  {len(test_df):,} teste")

    # 3. fit scaler no treino
    cols  = [c for c in df.columns if c not in SKIP_COLS]
    stats = fit_scaler(train_df, cols)

    scaler_out = {
        "_meta": {
            "variable":    variable,
            "start_year":  START_YEAR,
            "data_start":  data_start,
            "train_cutoff": cutoff,
            "n_train":     len(train_df),
            "n_test":      len(test_df),
            "train_ratio": TRAIN_RATIO,
            "min_stations": MIN_STATIONS,
        },
        **stats,
    }
    scaler_path = MODELS_DIR / f"1.6_scaler_{variable}.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_out, f, indent=2)
    print(f"  → {scaler_path.name}")

    # 4. transform e salva
    for split_df, split_name in [(train_df, "train"), (test_df, "test")]:
        scaled    = apply_scaler(split_df, stats, gpu_id=gpu_id)
        out_path  = DATA_DIR / f"{variable}_{split_name}_scaled.parquet"
        scaled.to_parquet(out_path, index=False, compression="snappy")
        size_mb   = out_path.stat().st_size / 1_048_576
        print(f"  → {out_path.name}   {size_mb:.0f} MB")

    elapsed = time.time() - t0
    print(f"  Concluído em {elapsed:.0f}s")


# ── worker para multiprocessing ───────────────────────────────────────────────

def _worker(args: tuple) -> None:
    variable, gpu_id = args
    process_variable(variable, gpu_id)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"=== 1.6 Split + Scale (train={TRAIN_RATIO:.0%} / test={1-TRAIN_RATIO:.0%}) ===")
    if HAS_CUPY:
        print(f"  GPU: CuPy disponível — {N_GPUS} GPUs | chunk={CHUNK_SIZE:,} linhas")
    else:
        print("  CPU: CuPy não encontrado — rodando em numpy")
    print()

    MODELS_DIR.mkdir(exist_ok=True)

    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    tasks     = [(v, i % N_GPUS) for i, v in enumerate(variables)]

    if HAS_CUPY and len(variables) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(len(variables), N_GPUS)) as pool:
            pool.map(_worker, tasks)
    else:
        for variable, gpu_id in tasks:
            process_variable(variable, gpu_id)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
