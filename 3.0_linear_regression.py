"""
3.0 — Regressão Linear: predição de measurement via vizinhos.

Para cada variável:
  1. Treina OLS (β = solve(XᵀX, Xᵀy)) no conjunto de treino escalado.
  2. Avalia no conjunto de TESTE escalado (mesmos arquivos que todos os
     modelos — comparação justa).
  3. Salva o modelo (coeficientes β) e o CSV de métricas por variável.

Features usadas (79 colunas):
    n01..n15   — medição dos vizinhos
    d01..d15   — distância em km
    a01..a15   — delta de altitude em m
    b01_sin..b15_sin, b01_cos..b15_cos — azimute sin/cos
    hour_sin, hour_cos, doy_sin, doy_cos — temporais cíclicos

Todos os slots n01..n15 são garantidos não-NaN pelo filtro do 1.6
(MIN_STATIONS = 15). Nenhum preenchimento de NaN é necessário.
Intercepto adicionado automaticamente. Métricas no espaço escalado.

ACELERAÇÃO GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Equação normal resolvida em chunks de CHUNK_SIZE linhas (~1.7 GB/chunk).
5 variáveis em paralelo em 5 GPUs via multiprocessing (spawn).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
    data/{variable}_train_scaled.parquet
    data/{variable}_test_scaled.parquet

Output:
    results/3.0_linear_regression/{variable}/model.npy   — coeficientes β
    results/3.0_linear_regression/{variable}/metrics.csv — métricas do teste
    results/3.0_linear_regression/metrics.csv            — resumo geral

Usage:
    python 3.0_linear_regression.py
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from utils import (
    RESULTS_DIR as _BASE_RESULTS,
    VARIABLES,
    compute_metrics,
    load_train,
    load_test,
    save_metrics,
)

RESULTS_DIR = _BASE_RESULTS / "3.0_linear_regression"

K          = 15
CHUNK_SIZE = 2_000_000
N_GPUS     = 5

VARIABLES_TO_RUN = None

NEIGHBOR_COLS = (
    [f"n{i+1:02d}" for i in range(K)]
    + [f"d{i+1:02d}" for i in range(K)]
    + [f"a{i+1:02d}" for i in range(K)]
    + [f"b{i+1:02d}_sin" for i in range(K)]
    + [f"b{i+1:02d}_cos" for i in range(K)]
)
TEMPORAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_COLS  = NEIGHBOR_COLS + TEMPORAL_COLS


# ── OLS — equação normal em chunks ────────────────────────────────────────────

def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if not HAS_CUPY:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    p   = X.shape[1]
    XtX = cp.zeros((p, p), dtype=cp.float64)
    Xty = cp.zeros(p,      dtype=cp.float64)

    for s in range(0, len(X), CHUNK_SIZE):
        Xc  = cp.asarray(X[s : s + CHUNK_SIZE])
        yc  = cp.asarray(y[s : s + CHUNK_SIZE])
        XtX += Xc.T @ Xc
        Xty += Xc.T @ yc

    return cp.asnumpy(cp.linalg.solve(XtX, Xty))


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if not HAS_CUPY:
        return X @ beta

    y_pred   = np.empty(len(X), dtype=np.float64)
    beta_gpu = cp.asarray(beta)
    for s in range(0, len(X), CHUNK_SIZE):
        Xc = cp.asarray(X[s : s + CHUNK_SIZE])
        y_pred[s : s + CHUNK_SIZE] = cp.asnumpy(Xc @ beta_gpu)
    return y_pred


# ── helpers ───────────────────────────────────────────────────────────────────

def build_X(df: pd.DataFrame) -> np.ndarray:
    raw = df[FEATURE_COLS].values.astype(np.float64)
    return np.hstack([np.ones((len(raw), 1), dtype=np.float64), raw])


# ── processamento por variável ────────────────────────────────────────────────

def process_variable(variable: str, gpu_id: int = 0) -> dict | None:
    try:
        train_df = load_train(variable)
        test_df  = load_test(variable)
    except FileNotFoundError as e:
        print(f"  SKIP {variable}: {e.filename} não encontrado — rode 1.6 primeiro.")
        return None

    if HAS_CUPY:
        cp.cuda.Device(gpu_id).use()

    prefix = f"[GPU {gpu_id}][{variable}]" if HAS_CUPY else f"[{variable}]"
    print(f"\n{prefix}")

    # treino
    X_train  = build_X(train_df)
    y_train  = train_df["measurement"].values.astype(np.float64)
    n_train  = len(train_df)
    print(f"  Treino : {n_train:,} registros  (X: {X_train.shape[0]:,} × {X_train.shape[1]})")

    print(f"  Ajustando OLS ...")
    beta = fit_ols(X_train, y_train)
    del train_df, X_train, y_train

    # teste
    X_test  = build_X(test_df)
    y_test  = test_df["measurement"].values.astype(np.float64)
    n_test  = len(test_df)
    print(f"  Teste  : {n_test:,} registros")

    print(f"  Calculando predições no teste ...")
    y_pred  = predict(X_test, beta)
    metrics = compute_metrics(y_test, y_pred)
    print(
        f"  MAE={metrics['MAE']:>8.4f}"
        f"  RMSE={metrics['RMSE']:>8.4f}"
        f"  R²={metrics['R²']:>7.4f}"
        f"  Bias={metrics['Bias']:>+8.4f}"
        f"  r={metrics['r']:>7.4f}"
    )

    # salva modelo e métricas por variável
    var_dir = RESULTS_DIR / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    np.save(var_dir / "model.npy", beta)

    out = save_metrics(metrics, RESULTS_DIR, variable, extra_cols={"n_train": n_train, "n_test": n_test})
    print(f"  → {var_dir}/model.npy + {out.name}")

    return {"variable": variable, "n_train": n_train, "n_test": n_test, **metrics}


# ── worker para multiprocessing ───────────────────────────────────────────────

def _worker(args: tuple) -> dict | None:
    variable, gpu_id = args
    return process_variable(variable, gpu_id)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== 3.0 Linear Regression ===")
    if HAS_CUPY:
        print(f"  GPU: CuPy disponível — {N_GPUS} GPUs | chunk={CHUNK_SIZE:,} linhas")
    else:
        print("  CPU: CuPy não encontrado — rodando em numpy")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    tasks     = [(v, i % N_GPUS) for i, v in enumerate(variables)]

    if HAS_CUPY and len(variables) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(len(variables), N_GPUS)) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [process_variable(v, gpu_id=0) for v, _ in tasks]

    rows = [r for r in results if r is not None]
    if not rows:
        print("\nNenhuma variável processada.")
        return

    metrics_df = pd.DataFrame(rows).set_index("variable").round(4)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv")

    print(f"\n=== Resumo (teste) ===")
    print(metrics_df.to_string())
    print(f"\n→ {RESULTS_DIR}/metrics.csv")


if __name__ == "__main__":
    main()
