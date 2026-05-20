"""
4.1 — Avaliação dos checkpoints salvos pelo 4.0_dense_layer no conjunto de teste.

Útil quando o treino completou mas a avaliação final crashou.
Carrega cada {variable}_{config}_dense.pt e gera metrics.csv sem re-treinar.

Usage:
    python 4.0_evaluate.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import (
    MODELS_DIR,
    RESULTS_DIR as _BASE_RESULTS,
    VARIABLES,
    compute_metrics,
    get_feature_cols,
    load_test,
    save_metrics,
)

RESULTS_DIR  = _BASE_RESULTS / "4.0_dense_layer"
FEATURE_COLS = get_feature_cols(k=15)
N_FEATURES   = len(FEATURE_COLS)
BATCH_SIZE   = 131_072

VARIABLES_TO_EVAL: list[str] | None = None   # None = todas; ou ex: ["temperature"]
CONFIGS_TO_EVAL:   list[str] | None = None   # None = todas; ou ex: ["base"]


# ── arquitetura (espelho do 4.0_dense_layer) ──────────────────────────────────

class DenseNet(nn.Module):
    def __init__(self, n_features: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for i, h in enumerate(hidden_dims):
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout / 2),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class Config:
    name: str
    hidden_dims: list[int]
    dropout: float


CONFIGS: list[Config] = [
    Config(name="base", hidden_dims=[256, 512, 256, 128, 64],           dropout=0.20),
    Config(name="wide", hidden_dims=[1024, 2048, 2048, 1024, 512],      dropout=0.15),
    Config(name="xl",   hidden_dims=[2048, 4096, 2048, 1024, 512],      dropout=0.10),
]


# ── avaliação ─────────────────────────────────────────────────────────────────

def evaluate(variable: str, cfg: Config, device: torch.device) -> dict | None:
    ckpt_path    = MODELS_DIR / f"{variable}_{cfg.name}_dense.pt"
    metrics_path = RESULTS_DIR / cfg.name / variable / "metrics.csv"

    if not ckpt_path.exists():
        print(f"  SKIP {variable}[{cfg.name}]: checkpoint não encontrado")
        return None

    if metrics_path.exists():
        print(f"  SKIP {variable}[{cfg.name}]: metrics.csv já existe")
        return None

    try:
        test_df = load_test(variable)
    except FileNotFoundError as e:
        print(f"  SKIP {variable}: {e.filename} não encontrado")
        return None

    print(f"  {variable}[{cfg.name}]...", flush=True)

    model = DenseNet(N_FEATURES, cfg.hidden_dims, cfg.dropout).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()

    X_test = torch.from_numpy(test_df[FEATURE_COLS].to_numpy(dtype=np.float32).copy())
    y_test = test_df["measurement"].to_numpy(dtype=np.float64)
    n_test = len(y_test)

    use_amp = device.type == "cuda"
    preds: list[np.ndarray] = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for s in range(0, len(X_test), BATCH_SIZE * 4):
            chunk = X_test[s : s + BATCH_SIZE * 4].to(device)
            preds.append(model(chunk).cpu().numpy())
    y_pred = np.concatenate(preds).astype(np.float64)

    metrics = compute_metrics(y_test, y_pred)
    print(
        f"    MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}"
        f"  R²={metrics['R²']:.4f}  Bias={metrics['Bias']:+.4f}  r={metrics['r']:.4f}"
    )

    save_metrics(
        metrics, RESULTS_DIR / cfg.name, variable,
        extra_cols={"config": cfg.name, "n_test": n_test},
    )
    return {"config": cfg.name, "variable": variable, **metrics}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== 4.0 Evaluate — avaliação no teste ===\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}\n")

    variables = VARIABLES_TO_EVAL or VARIABLES
    cfg_names = set(CONFIGS_TO_EVAL or [c.name for c in CONFIGS])
    configs   = [c for c in CONFIGS if c.name in cfg_names]

    rows: list[dict] = []
    for variable in variables:
        for cfg in configs:
            r = evaluate(variable, cfg, device)
            if r:
                rows.append(r)

    if not rows:
        print("\nNenhuma avaliação realizada.")
        return

    df = pd.DataFrame(rows).round(4)
    print(f"\n=== Resumo ===")
    for cfg_name, grp in df.groupby("config"):
        print(f"\n  [{cfg_name}]")
        print(grp.set_index("variable")[["MAE", "RMSE", "R²", "r"]].to_string())

    if df["config"].nunique() > 1:
        comparison = df.pivot_table(
            index="variable", columns="config", values=["MAE", "RMSE", "R²"]
        ).round(4)
        comparison.to_csv(RESULTS_DIR / "comparison.csv")
        print(f"\n→ {RESULTS_DIR}/comparison.csv")


if __name__ == "__main__":
    main()
