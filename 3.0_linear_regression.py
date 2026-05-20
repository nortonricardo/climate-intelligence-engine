"""
3.0 — Ridge Regression: predição de measurement via vizinhos.

Para cada variável:
  1. Carrega treino completo em memória (servidor 512 GB — sem chunking).
  2. Separa últimos 10% do treino como validação temporal.
  3. Computa XᵀX e Xᵀy uma única vez.
  4. Busca o melhor λ (ALPHAS) resolvendo Ridge = solve(XᵀX + λI, Xᵀy)
     para cada candidato e avaliando MAE na validação — custo ~zero (80×80).
  5. Re-treina com o melhor λ sobre treino + validação completos.
  6. Avalia no teste e salva modelo e métricas.

Ridge vs OLS puro:
  n01..n15 são altamente correlacionados (temperaturas de estações vizinhas),
  tornando XᵀX mal-condicionada. Ridge adiciona λI à diagonal, estabilizando
  o solve e controlando a multicolinearidade sem sacrificar precisão.

Features (79 colunas):
    n01..n15          — medição dos vizinhos      (MinMax [0,1])
    d01..d15          — distância em km           (MinMax [0,1])
    a01..a15          — delta de altitude em m    (MinMax [0,1])
    b01..b15 sin/cos  — azimute                   ([−1,1], sem scaler)
    hour/doy sin/cos  — temporais cíclicos        ([−1,1], sem scaler)

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

RESULTS_DIR      = _BASE_RESULTS / "3.0_linear_regression"
VARIABLES_TO_RUN = None   # None = todas; ou ex: ["temperature"]
VAL_RATIO        = 0.10   # últimos 10% do treino → validação para busca de λ
FEATURE_COLS     = get_feature_cols(k=15)

# candidatos de λ para Ridge — escala log cobre desde quase-OLS até forte regularização
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# ── Ridge via equação normal ──────────────────────────────────────────────────

def compute_normal_equations(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (XᵀX, Xᵀy) — base para qualquer λ de Ridge."""
    return X.T @ X, X.T @ y


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, alpha: float) -> np.ndarray:
    """β = solve(XᵀX + αI, Xᵀy) — não modifica XᵀX original."""
    p      = XtX.shape[0]
    XtX_r  = XtX + alpha * np.eye(p, dtype=np.float64)
    return np.linalg.solve(XtX_r, Xty)


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

    # split temporal: últimos VAL_RATIO do treino → validação para busca de λ
    n_val    = max(1, int(len(train_df) * VAL_RATIO))
    val_df   = train_df.iloc[-n_val:].copy()
    fit_df   = train_df.iloc[:-n_val].copy()
    n_train  = len(train_df)
    _step(f"treino={len(fit_df):,}  val={n_val:,}  teste={len(test_df):,}")

    # monta matrizes de features
    X_fit = build_X(fit_df);  y_fit = fit_df["measurement"].to_numpy(dtype=np.float64);  del fit_df
    X_val = build_X(val_df);  y_val = val_df["measurement"].to_numpy(dtype=np.float64);  del val_df

    _step(f"computando XᵀX  (X: {X_fit.shape[0]:,} × {X_fit.shape[1]})...")
    XtX, Xty = compute_normal_equations(X_fit, y_fit)

    # busca do melhor λ — cada solve é 80×80, custo negligível
    _step(f"buscando melhor λ em {ALPHAS}...")
    best_alpha, best_val_mae = ALPHAS[0], float("inf")
    for alpha in ALPHAS:
        beta    = ridge_solve(XtX, Xty, alpha)
        val_mae = float(np.mean(np.abs(X_val @ beta - y_val)))
        marker  = "  ←" if val_mae < best_val_mae else ""
        _step(f"  λ={alpha:<8}  val_mae={val_mae:.6f}{marker}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha   = alpha

    _step(f"melhor λ={best_alpha}  val_mae={best_val_mae:.6f}")

    # re-treina com o melhor λ sobre todo o treino (fit + val)
    _step("re-treinando com treino completo + melhor λ...")
    X_full = build_X(train_df);  y_full = train_df["measurement"].to_numpy(dtype=np.float64);  del train_df
    XtX_full, Xty_full = compute_normal_equations(X_full, y_full);  del X_full, y_full
    beta = ridge_solve(XtX_full, Xty_full, best_alpha)

    # avalia no teste
    X_test = build_X(test_df);  y_test = test_df["measurement"].to_numpy(dtype=np.float64);  del test_df
    n_test  = len(X_test)
    y_pred  = X_test @ beta;  del X_test

    metrics = compute_metrics(y_test, y_pred)
    _step(
        f"MAE={metrics['MAE']:.4f}"
        f"  RMSE={metrics['RMSE']:.4f}"
        f"  R²={metrics['R²']:.4f}"
        f"  Bias={metrics['Bias']:+.4f}"
        f"  r={metrics['r']:.4f}"
    )

    var_dir = RESULTS_DIR / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    np.save(var_dir / "model.npy", beta)

    out = save_metrics(
        metrics, RESULTS_DIR, variable,
        extra_cols={"n_train": n_train, "n_test": n_test, "best_alpha": best_alpha, "best_val_mae": round(best_val_mae, 6)},
    )
    _step(f"→ model.npy + {out.name}  ({time.time() - t0:.0f}s total)")

    return {"variable": variable, "n_train": n_train, "n_test": n_test, "best_alpha": best_alpha, **metrics}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"=== 3.0 Ridge Regression  λ candidatos={ALPHAS} ===\n")
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
