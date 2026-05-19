"""
3.0 — Regressão Linear: predição de measurement via vizinhos.

Para cada variável:
  1. Treina OLS (β = solve(XᵀX, Xᵀy)) no conjunto de treino escalado.
  2. Avalia no conjunto de TESTE escalado (comparação justa com 2.0+).
  3. Salva os coeficientes β e o CSV de métricas por variável.

Features (79 colunas):
    n01..n15                       — medição dos vizinhos (MinMax [0,1])
    d01..d15                       — distância em km    (MinMax [0,1])
    a01..a15                       — delta de altitude  (MinMax [0,1])
    b01_sin..b15_sin/cos           — azimute sin/cos    ([−1,1], sem scaler)
    hour_sin, hour_cos             — hora cíclica       ([−1,1], sem scaler)
    doy_sin,  doy_cos              — dia do ano cíclico ([−1,1], sem scaler)

n01..n15 são garantidamente não-NaN pelo filtro do 1.6. Intercepto
adicionado automaticamente. Métricas avaliadas no espaço escalado.

OLS via equação normal acumulada em chunks para evitar pico de memória
com datasets de dezenas de milhões de linhas.

Input:
    data/{variable}_train_scaled.parquet
    data/{variable}_test_scaled.parquet

Output:
    results/3.0_linear_regression/{variable}/model.npy   — coeficientes β
    results/3.0_linear_regression/{variable}/metrics.csv
    results/3.0_linear_regression/metrics.csv            — resumo geral

Usage:
    python 3.0_linear_regression.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from utils import (
    RESULTS_DIR as _BASE_RESULTS,
    VARIABLES,
    compute_metrics,
    get_feature_cols,
    load_train,
    load_test,
    save_metrics,
)

RESULTS_DIR  = _BASE_RESULTS / "3.0_linear_regression"
CHUNK_SIZE   = 2_000_000   # linhas por chunk para acumulação XᵀX
VARIABLES_TO_RUN = None    # None = todas; ou ex: ["temperature"]
FEATURE_COLS = get_feature_cols(k=15)


# ── OLS via equação normal em chunks ─────────────────────────────────────────

def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """β = solve(XᵀX, Xᵀy) acumulado em chunks — evita matriz intermediária enorme."""
    p   = X.shape[1]
    XtX = np.zeros((p, p), dtype=np.float64)
    Xty = np.zeros(p,      dtype=np.float64)
    for s in range(0, len(X), CHUNK_SIZE):
        Xc   = X[s : s + CHUNK_SIZE]
        yc   = y[s : s + CHUNK_SIZE]
        XtX += Xc.T @ Xc
        Xty += Xc.T @ yc
    return np.linalg.solve(XtX, Xty)


# ── helpers ───────────────────────────────────────────────────────────────────

def build_X(df: pd.DataFrame) -> np.ndarray:
    raw = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    return np.hstack([np.ones((len(raw), 1), dtype=np.float64), raw])


# ── processamento por variável ────────────────────────────────────────────────

def process_variable(variable: str) -> dict | None:
    try:
        train_df = load_train(variable)
        test_df  = load_test(variable)
    except FileNotFoundError as e:
        print(f"  SKIP {variable}: {e.filename} não encontrado — rode 1.6 primeiro.")
        return None

    def _step(msg: str) -> None:
        print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)

    t0 = time.time()
    print(f"\n[{variable}]", flush=True)

    # treino
    X_train = build_X(train_df)
    y_train = train_df["measurement"].to_numpy(dtype=np.float64)
    n_train = len(train_df)
    del train_df
    _step(f"treino: {n_train:,} registros  (X: {X_train.shape[0]:,} × {X_train.shape[1]})")

    _step("ajustando OLS...")
    beta = fit_ols(X_train, y_train)
    del X_train, y_train
    _step("OLS concluído")

    # teste
    X_test = build_X(test_df)
    y_test = test_df["measurement"].to_numpy(dtype=np.float64)
    n_test = len(test_df)
    del test_df
    _step(f"teste : {n_test:,} registros")

    y_pred  = X_test @ beta
    del X_test
    metrics = compute_metrics(y_test, y_pred)
    _step(
        f"MAE={metrics['MAE']:>8.4f}"
        f"  RMSE={metrics['RMSE']:>8.4f}"
        f"  R²={metrics['R²']:>7.4f}"
        f"  Bias={metrics['Bias']:>+8.4f}"
        f"  r={metrics['r']:>7.4f}"
    )

    var_dir = RESULTS_DIR / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    np.save(var_dir / "model.npy", beta)

    out = save_metrics(metrics, RESULTS_DIR, variable, extra_cols={"n_train": n_train, "n_test": n_test})
    _step(f"→ model.npy + {out.name}  ({time.time() - t0:.0f}s total)")

    return {"variable": variable, "n_train": n_train, "n_test": n_test, **metrics}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== 3.0 Linear Regression (CPU — equação normal) ===\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    results   = [process_variable(v) for v in variables]

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
