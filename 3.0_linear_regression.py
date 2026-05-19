"""
3.0 — Regressão Linear: predição de measurement via vizinhos.

Para cada variável:
  1. Determina training_start: primeira data em que ≥ MIN_STATIONS estações
     vizinhas têm dado disponível (ou seja, n01..n{MIN_STATIONS} não-NaN).
  2. Treina OLS (β = (XᵀX)⁻¹Xᵀy) com os dados a partir de training_start.
  3. Prediz measurement para todo o conjunto de treino.
  4. Calcula métricas: MAE, RMSE, R², Bias, r — todas implementadas manualmente.
  5. Salva resultado em results/3.0_linear_regression_{variable}.csv com as
     mesmas colunas do arquivo _neighbors mais as colunas `prediction` e
     `training_start`.

Features usadas (104 colunas):
    n01..n20   — medição dos vizinhos
    d01..d20   — distância em km
    a01..a20   — delta de altitude em m
    b01_sin..b20_sin, b01_cos..b20_cos — azimute sin/cos
    hour_sin, hour_cos, doy_sin, doy_cos — temporais cíclicos

Colunas com NaN são substituídas por 0 antes do ajuste (sem dado = sem
contribuição). Um intercepto é adicionado automaticamente.

Input:
    data/{variable}_neighbors.parquet

Output:
    results/3.0_linear_regression_{variable}.csv
    results/3.0_linear_regression_metrics.csv

Usage:
    python 3.0_linear_regression.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

K              = 20
MIN_STATIONS   = 15   # limiar de vizinhos disponíveis para treino
VARIABLES_TO_RUN = None  # None = todas; ou ex: ["temperature"]

VARIABLES = [
    "temperature",
    "humidity",
    "rainfall",
    "global_radiation",
    "pressure",
]

NEIGHBOR_COLS = (
    [f"n{i+1:02d}" for i in range(K)]
    + [f"d{i+1:02d}" for i in range(K)]
    + [f"a{i+1:02d}" for i in range(K)]
    + [f"b{i+1:02d}_sin" for i in range(K)]
    + [f"b{i+1:02d}_cos" for i in range(K)]
)
TEMPORAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_COLS  = NEIGHBOR_COLS + TEMPORAL_COLS   # 104 colunas


# ── métricas (numpy puro) ─────────────────────────────────────────────────────

def _pearsonr(y: np.ndarray, yhat: np.ndarray) -> float:
    y_m    = y    - y.mean()
    yhat_m = yhat - yhat.mean()
    denom  = np.sqrt(np.sum(y_m ** 2)) * np.sqrt(np.sum(yhat_m ** 2))
    return float(np.sum(y_m * yhat_m) / denom) if denom > 0 else np.nan


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_true - y_pred
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((y_true - y_true.mean()) ** 2)

    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    bias = float(np.mean(residuals))
    r    = _pearsonr(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R²": r2, "Bias": bias, "r": r}


# ── regressão OLS (numpy puro) ────────────────────────────────────────────────

def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Retorna β = (XᵀX)⁻¹Xᵀy via pseudo-inversa (lstsq)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ── data de início de treino ──────────────────────────────────────────────────

def find_training_start(df: pd.DataFrame, min_stations: int) -> str | None:
    """
    Retorna a primeira data (string ISO) em que pelo menos min_stations
    das colunas n01..n{min_stations} são não-NaN na mesma linha.

    Percorre em ordem crescente de time. Não assume que todas as linhas de
    uma mesma data são iguais: usa a primeira linha onde a condição é
    satisfeita.
    """
    threshold_cols = [f"n{i+1:02d}" for i in range(min_stations)]
    valid_count    = df[threshold_cols].notna().sum(axis=1)
    mask           = valid_count >= min_stations
    if not mask.any():
        return None
    first_ts = df.loc[mask, "time"].min()
    return str(first_ts.date())


# ── processamento por variável ────────────────────────────────────────────────

def process_variable(variable: str) -> dict | None:
    path = DATA_DIR / f"{variable}_neighbors.parquet"
    if not path.exists():
        print(f"  SKIP {variable}: {path.name} não encontrado — rode 1.5 primeiro.")
        return None

    print(f"\n[{variable}]")
    df = pd.read_parquet(path)

    # garante que time é datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    # 1. Determina training_start
    training_start = find_training_start(df, MIN_STATIONS)
    if training_start is None:
        print(f"  SKIP {variable}: nunca alcança {MIN_STATIONS} vizinhos disponíveis.")
        return None
    print(f"  training_start = {training_start}  (mínimo {MIN_STATIONS} vizinhos)")

    train_df = df[df["time"] >= training_start].copy()
    n_train  = len(train_df)
    print(f"  Registros para treino : {n_train:,}")

    # 2. Monta matriz de features (NaN → 0, sem dado = sem contribuição)
    X_raw = train_df[FEATURE_COLS].values.astype(np.float64)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # adiciona coluna de intercepto
    X = np.hstack([np.ones((X_raw.shape[0], 1), dtype=np.float64), X_raw])
    y = train_df["measurement"].values.astype(np.float64)

    # 3. Treino OLS
    print(f"  Ajustando OLS  (X: {X.shape[0]:,} × {X.shape[1]}) ...")
    beta = fit_ols(X, y)

    # 4. Predição e métricas
    y_pred = X @ beta
    metrics = compute_metrics(y, y_pred)
    print(
        f"  MAE={metrics['MAE']:>8.4f}"
        f"  RMSE={metrics['RMSE']:>8.4f}"
        f"  R²={metrics['R²']:>7.4f}"
        f"  Bias={metrics['Bias']:>+8.4f}"
        f"  r={metrics['r']:>7.4f}"
    )

    # 5. Salva CSV com todas as colunas dos vizinhos + prediction + training_start
    out_df = train_df.copy()
    out_df["prediction"]    = np.round(y_pred, 4)
    out_df["training_start"] = training_start

    out_path = RESULTS_DIR / f"3.0_linear_regression_{variable}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  → {out_path.name}  ({n_train:,} linhas)")

    return {"variable": variable, "training_start": training_start, "n": n_train, **metrics}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"=== 3.0 Linear Regression (min_stations={MIN_STATIONS}) ===\n")
    RESULTS_DIR.mkdir(exist_ok=True)

    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    rows = []
    for variable in variables:
        result = process_variable(variable)
        if result is not None:
            rows.append(result)

    if not rows:
        print("\nNenhuma variável processada.")
        return

    metrics_df = (
        pd.DataFrame(rows)
        .set_index("variable")
        .round(4)
    )
    metrics_path = RESULTS_DIR / "3.0_linear_regression_metrics.csv"
    metrics_df.to_csv(metrics_path)

    print(f"\n=== Resumo ===")
    print(metrics_df.to_string())
    print(f"\n→ {metrics_path}")


if __name__ == "__main__":
    main()
