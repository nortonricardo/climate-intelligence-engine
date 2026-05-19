"""
2.0 — Neighbors: baseline vizinho mais próximo (n01 vs measurement).

Avalia o vizinho mais próximo (n01) como estimador da medição da
estação-alvo no conjunto de TESTE. Serve como baseline mínimo —
qualquer modelo treinado deve superar essas métricas.

Métricas calculadas no espaço MinMax [0, 1] para comparação direta
com os modelos treinados (3.0+). n01..n15 são garantidamente não-NaN
pelo filtro do 1.6.

Métricas por variável:
    n        — registros de teste
    MAE      — Erro absoluto médio
    RMSE     — Raiz do erro quadrático médio
    R²       — Coeficiente de determinação
    Bias     — Erro sistemático (positivo: n01 subestima o alvo)
    r        — Correlação de Pearson

Input:
    data/{variable}_test_scaled.parquet

Output:
    results/2.0_neighbors/metrics.csv

Usage:
    python 2.0_neighbors.py
"""

import numpy as np
import pandas as pd

from utils import VARIABLES, compute_metrics, load_test, RESULTS_DIR as _BASE_RESULTS

RESULTS_DIR = _BASE_RESULTS / "2.0_neighbors"


def main() -> None:
    print("=== 2.0 Neighbors — baseline (n01 vs measurement, teste) ===\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for variable in VARIABLES:
        try:
            df = load_test(variable, columns=["measurement", "n01"])
        except FileNotFoundError:
            print(f"  SKIP {variable}: {variable}_test_scaled.parquet não encontrado — rode 1.6 primeiro.")
            continue

        n = len(df)
        metrics = compute_metrics(
            df["measurement"].values.astype(np.float64),
            df["n01"].values.astype(np.float64),
        )

        rows.append({"variable": variable, "n": n, **metrics})

        print(
            f"  {variable:<20}"
            f"  n={n:>12,}"
            f"  MAE={metrics['MAE']:>8.4f}"
            f"  RMSE={metrics['RMSE']:>8.4f}"
            f"  R²={metrics['R²']:>7.4f}"
            f"  Bias={metrics['Bias']:>+8.4f}"
            f"  r={metrics['r']:>7.4f}"
        )

    if not rows:
        print("Nenhum arquivo encontrado.")
        return

    out = RESULTS_DIR / "metrics.csv"
    pd.DataFrame(rows).set_index("variable").round(4).to_csv(out)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
