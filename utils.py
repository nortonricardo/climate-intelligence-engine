"""
utils.py — funções compartilhadas entre os scripts do pipeline.

Métricas (numpy puro, sem sklearn/scipy):
    mae, rmse, r2, bias, pearsonr, compute_metrics

Features:
    get_feature_cols(k)  — lista de colunas de features para k vizinhos

I/O:
    load_train, load_test, load_train_test
    save_metrics
    load_scaler

Paths centralizados:
    DATA_DIR, MODELS_DIR, RESULTS_DIR
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"

VARIABLES = [
    "temperature",
    "humidity",
    "rainfall",
    "global_radiation",
    "pressure",
]


# ── métricas individuais ──────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true - y_pred))


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_m    = y_true - y_true.mean()
    yhat_m = y_pred - y_pred.mean()
    denom  = np.sqrt(np.sum(y_m ** 2)) * np.sqrt(np.sum(yhat_m ** 2))
    return float(np.sum(y_m * yhat_m) / denom) if denom > 0 else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R²":   r2(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "r":    pearsonr(y_true, y_pred),
    }


# ── features ─────────────────────────────────────────────────────────────────

def get_feature_cols(k: int = 15) -> list[str]:
    """
    Retorna a lista ordenada de colunas de features para k vizinhos.
    Ordem: n, d, a, b_sin, b_cos, temporais cíclicos.
    """
    return (
        [f"n{i+1:02d}"     for i in range(k)]
        + [f"d{i+1:02d}"   for i in range(k)]
        + [f"a{i+1:02d}"   for i in range(k)]
        + [f"b{i+1:02d}_sin" for i in range(k)]
        + [f"b{i+1:02d}_cos" for i in range(k)]
        + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    )


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_train(variable: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / f"{variable}_train_scaled.parquet"
    return pd.read_parquet(path, columns=columns)


def load_test(variable: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / f"{variable}_test_scaled.parquet"
    return pd.read_parquet(path, columns=columns)


def load_train_test(
    variable: str,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_train(variable, columns), load_test(variable, columns)


def load_scaler(variable: str) -> tuple:
    """
    Carrega o scaler salvo pelo 1.6.
    Retorna (scaler, cols) onde scaler é o MinMaxScaler e cols são as colunas escaladas.
    """
    path   = MODELS_DIR / f"scaler_{variable}.scaler"
    bundle = joblib.load(path)
    return bundle["scaler"], bundle["cols"]


def save_metrics(
    metrics: dict,
    results_dir: Path,
    variable: str,
    *,
    extra_cols: dict | None = None,
) -> Path:
    """
    Salva metrics em results_dir/{variable}/metrics.csv.
    extra_cols: colunas adicionais a incluir na linha (ex: n_train, n_test).
    Retorna o path do arquivo salvo.
    """
    row     = {**(extra_cols or {}), **metrics}
    var_dir = results_dir / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    out = var_dir / "metrics.csv"
    pd.DataFrame([row]).round(4).to_csv(out, index=False)
    return out
