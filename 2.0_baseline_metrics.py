"""
2.0 — Baseline metrics: vizinho mais próximo (n01) vs measurement.

Avalia o vizinho mais próximo disponível (n01) como estimador da
medição da estação-alvo. Serve como baseline mínimo de gap-filling —
qualquer modelo treinado deve superar essas métricas.

Métricas calculadas por variável:
    n        — número de registros com n01 disponível
    n_nan    — registros sem nenhum vizinho disponível
    MAE      — Erro absoluto médio (mesma unidade da variável)
    RMSE     — Raiz do erro quadrático médio
    R²       — Coeficiente de determinação
    Bias     — Erro sistemático — positivo: n01 subestima o alvo
    r        — Correlação de Pearson

Input:
    data/{variable}_neighbors.parquet

Output:
    results/2.0_baseline_metrics.csv

Usage:
    python 2.0_baseline_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

VARIABLES = [
    "temperature",
    "humidity",
    "rainfall",
    "global_radiation",
    "pressure",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_true - y_pred
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((y_true - y_true.mean()) ** 2)

    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    bias = float(np.mean(residuals))
    y_m    = y_true - y_true.mean()
    yhat_m = y_pred - y_pred.mean()
    denom  = np.sqrt(np.sum(y_m ** 2)) * np.sqrt(np.sum(yhat_m ** 2))
    r      = float(np.sum(y_m * yhat_m) / denom) if denom > 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "R²": r2, "Bias": bias, "r": r}


def main() -> None:
    print("=== 2.0 Baseline Metrics (n01 vs measurement) ===\n")
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for variable in VARIABLES:
        path = DATA_DIR / f"{variable}_neighbors.parquet"
        if not path.exists():
            print(f"  SKIP {variable}: {path.name} não encontrado — rode 1.5 primeiro.")
            continue

        df = pd.read_parquet(path, columns=["measurement", "n01"])

        n_total = len(df)
        n_nan   = int(df["n01"].isna().sum())
        df      = df.dropna(subset=["n01"])
        n_valid = len(df)

        metrics = compute_metrics(
            df["measurement"].values.astype(np.float64),
            df["n01"].values.astype(np.float64),
        )

        rows.append({"variable": variable, "n": n_valid, "n_nan": n_nan, **metrics})

        print(
            f"  {variable:<20}"
            f"  n={n_valid:>12,}"
            f"  MAE={metrics['MAE']:>8.4f}"
            f"  RMSE={metrics['RMSE']:>8.4f}"
            f"  R²={metrics['R²']:>7.4f}"
            f"  Bias={metrics['Bias']:>+8.4f}"
            f"  r={metrics['r']:>7.4f}"
        )

    if not rows:
        print("Nenhum arquivo encontrado.")
        return

    result = (
        pd.DataFrame(rows)
        .set_index("variable")
        .round(4)
    )

    out = RESULTS_DIR / "2.0_baseline_metrics.csv"
    result.to_csv(out)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
