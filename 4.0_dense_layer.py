"""
4.0 — Rede Neural Densa (MLP) para gap-filling.

Arquitetura:
    Input(79) → [Linear → BN → GELU → Dropout] × 4 → Linear(1)
    Camadas: 256 → 512 → 256 → 128 → 64 → 1

Treinamento:
    Otimizador  : AdamW (lr=2e-3, weight_decay=1e-4)
    Loss        : Huber (delta=0.05, robusto a outliers no espaço [0,1])
    Scheduler   : ReduceLROnPlateau (fator=0.5, patience=5)
    Early stop  : patience=25 épocas sem melhora no MAE de validação
    Val split   : últimos 10% do treino (temporal — sem leakage)
    Precisão    : float32 + AMP (mixed precision) quando GPU disponível

GPU:
    Detecta automaticamente as GPUs disponíveis.
    Com múltiplas GPUs e múltiplas variáveis: uma variável por GPU em paralelo.
    Sem GPU: fallback para CPU.

Input:
    data/{variable}_train_scaled.parquet
    data/{variable}_test_scaled.parquet

Output:
    models/{variable}_dense.pt              — melhor state_dict (menor val MAE)
    results/4.0_dense_layer/{variable}/training_log.csv  — loss/MAE por época
    results/4.0_dense_layer/{variable}/metrics.csv       — métricas no teste
    results/4.0_dense_layer/metrics.csv                  — resumo geral

Usage:
    python 4.0_dense_layer.py
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    MODELS_DIR,
    RESULTS_DIR as _BASE_RESULTS,
    VARIABLES,
    compute_metrics,
    get_feature_cols,
    load_test,
    load_train,
    save_metrics,
)

torch.backends.cudnn.benchmark = True

# ── hiperparâmetros ───────────────────────────────────────────────────────────

HIDDEN_DIMS  = [256, 512, 256, 128, 64]
DROPOUT      = 0.2
BATCH_SIZE   = 65_536
LR           = 2e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 750
PATIENCE     = 25          # épocas sem melhora → early stop
VAL_RATIO    = 0.10        # fração final do treino usada como validação
HUBER_DELTA  = 0.05        # loss Huber no espaço [0,1]

VARIABLES_TO_RUN = None    # None = todas; ou ex: ["temperature"]
RESULTS_DIR      = _BASE_RESULTS / "4.0_dense_layer"
FEATURE_COLS     = get_feature_cols(k=15)
N_FEATURES       = len(FEATURE_COLS)   # 79


# ── GPU detection ─────────────────────────────────────────────────────────────

def available_devices() -> list[str]:
    if torch.cuda.is_available():
        devs = [f"cuda:{i}" for i in range(min(2, torch.cuda.device_count()))]
        print(f"  GPUs detectadas: {len(devs)}")
        for d in devs:
            name = torch.cuda.get_device_name(d)
            mem  = torch.cuda.get_device_properties(d).total_memory / 1024**3
            print(f"    {d}  {name}  ({mem:.1f} GB)")
        return devs
    print("  Nenhuma GPU detectada — usando CPU")
    return ["cpu"]


# ── arquitetura ───────────────────────────────────────────────────────────────
#
#  Input  (79)
#    │
#    ├─ Linear(79 → 256)  → BatchNorm1d → GELU → Dropout(0.20)  ← expande representação
#    ├─ Linear(256 → 512) → BatchNorm1d → GELU → Dropout(0.20)  ← pico de capacidade
#    ├─ Linear(512 → 256) → BatchNorm1d → GELU → Dropout(0.20)  ← comprime
#    ├─ Linear(256 → 128) → BatchNorm1d → GELU → Dropout(0.20)
#    ├─ Linear(128 → 64)  → BatchNorm1d → GELU → Dropout(0.10)  ← dropout reduzido na última hidden
#    └─ Linear(64 → 1)    ← saída escalar, sem ativação (regressão)
#
#  BatchNorm: estabiliza gradientes em datasets grandes e acelera convergência.
#  GELU: suave e diferenciável em x=0; supera ReLU em modelos profundos.
#  Dropout: regularização — reduzido na última camada hidden para não perder
#           representações já comprimidas.

class DenseNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
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


# ── dataset helpers ───────────────────────────────────────────────────────────

def df_to_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(df[FEATURE_COLS].to_numpy(dtype=np.float32).copy())
    y = torch.from_numpy(df["measurement"].to_numpy(dtype=np.float32).copy())
    return X, y


# ── treinamento ───────────────────────────────────────────────────────────────

def train_variable(variable: str, device_str: str) -> dict | None:
    device = torch.device(device_str)
    use_amp = device.type == "cuda"

    def _step(msg: str) -> None:
        print(f"  [{variable:<18s} {time.time() - t0:6.1f}s] {msg}", flush=True)

    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"\n[{variable}] → {device_str}", flush=True)

    # ── carrega dados ────────────────────────────────────────────────────────
    try:
        train_df = load_train(variable)
        test_df  = load_test(variable)
    except FileNotFoundError as e:
        print(f"  SKIP {variable}: {Path(e.filename).name} não encontrado — rode 1.6.")
        return None

    # split temporal: últimos VAL_RATIO do treino → validação
    n_val    = max(1, int(len(train_df) * VAL_RATIO))
    val_df   = train_df.iloc[-n_val:].copy()
    train_df = train_df.iloc[:-n_val].copy()

    _step(
        f"treino={len(train_df):,}  val={len(val_df):,}  teste={len(test_df):,}"
    )

    X_train, y_train = df_to_tensors(train_df); del train_df
    X_val,   y_val   = df_to_tensors(val_df);   del val_df

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
        drop_last=True,
    )

    # ── modelo ────────────────────────────────────────────────────────────────
    model     = torch.compile(DenseNet(N_FEATURES, HIDDEN_DIMS, DROPOUT).to(device))
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if use_amp:
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
        mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        _step(f"modelo: {n_params:,} params  |  GPU mem: {mem_alloc:.2f}/{mem_total:.1f} GB")

    # ── loop de épocas ────────────────────────────────────────────────────────
    var_dir    = RESULTS_DIR / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = MODELS_DIR / f"{variable}_dense.pt"
    MODELS_DIR.mkdir(exist_ok=True)

    best_val_mae  = float("inf")
    no_improve    = 0
    log_rows: list[dict] = []
    epoch_times: list[float] = []

    X_val_dev = X_val.to(device)
    y_val_np  = y_val.numpy()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        # treino
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(X_b)
                loss = criterion(pred, y_b)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        # validação
        model.eval()
        val_chunks: list[np.ndarray] = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for vs in range(0, len(X_val_dev), BATCH_SIZE * 4):
                val_chunks.append(model(X_val_dev[vs : vs + BATCH_SIZE * 4]).cpu().numpy())
        val_pred = np.concatenate(val_chunks)
        val_mae  = float(np.mean(np.abs(val_pred - y_val_np)))
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val_np) ** 2)))

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_dt = time.time() - epoch_start
        epoch_times.append(epoch_dt)
        avg_dt  = sum(epoch_times[-10:]) / len(epoch_times[-10:])
        eta_s   = avg_dt * (MAX_EPOCHS - epoch)
        eta_str = (f"{int(eta_s // 3600)}h{int(eta_s % 3600 // 60):02d}m"
                   if eta_s >= 3600 else f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s")

        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_mae": round(val_mae, 6), "val_rmse": round(val_rmse, 6),
            "lr": current_lr,
        })

        # checkpoint do melhor modelo
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve   = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            _step(
                f"epoch {epoch:4d}/{MAX_EPOCHS}  loss={train_loss:.5f}"
                f"  val_mae={val_mae:.5f}  best={best_val_mae:.5f}"
                f"  lr={current_lr:.2e}  Δt={epoch_dt:.1f}s  eta={eta_str}"
                + ("  ✓" if no_improve == 0 else f"  ({no_improve}/{PATIENCE})")
            )

        if no_improve >= PATIENCE:
            _step(f"early stop na época {epoch} — sem melhora por {PATIENCE} épocas")
            break

    # ── log de treinamento ────────────────────────────────────────────────────
    pd.DataFrame(log_rows).to_csv(var_dir / "training_log.csv", index=False)

    # ── avalia no teste com o melhor modelo ───────────────────────────────────
    _step("avaliando no teste com o melhor modelo...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    X_test, y_test = df_to_tensors(test_df); del test_df
    preds: list[np.ndarray] = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for s in range(0, len(X_test), BATCH_SIZE * 4):
            chunk = X_test[s : s + BATCH_SIZE * 4].to(device)
            preds.append(model(chunk).cpu().numpy())
    y_pred   = np.concatenate(preds)
    y_test_np = y_test.numpy()

    metrics = compute_metrics(y_test_np.astype(np.float64), y_pred.astype(np.float64))
    n_train_final = len(X_train)
    n_test        = len(X_test)

    _step(
        f"MAE={metrics['MAE']:.4f}"
        f"  RMSE={metrics['RMSE']:.4f}"
        f"  R²={metrics['R²']:.4f}"
        f"  Bias={metrics['Bias']:+.4f}"
        f"  r={metrics['r']:.4f}"
    )

    out = save_metrics(
        metrics, RESULTS_DIR, variable,
        extra_cols={"n_train": n_train_final, "n_test": n_test, "best_val_mae": round(best_val_mae, 6)},
    )
    _step(f"→ {ckpt_path.name} + {out.name}  ({time.time() - t0:.0f}s total)")

    return {"variable": variable, "n_train": n_train_final, "n_test": n_test, **metrics}


# ── worker para multiprocessing ───────────────────────────────────────────────

def _worker(args: tuple) -> dict | None:
    variable, device_str = args
    torch.set_num_threads(8)
    return train_variable(variable, device_str)


# ── main ──────────────────────────────────────────────────────────────────────

REQUIRE_GPU = True   # False → permite fallback para CPU


def main() -> None:
    print("=== 4.0 Dense Layer (MLP) ===\n")
    print(f"  arch={HIDDEN_DIMS}  dropout={DROPOUT}")
    print(f"  batch={BATCH_SIZE}  lr={LR}  patience={PATIENCE}  max_epochs={MAX_EPOCHS}\n")

    devices = available_devices()
    if REQUIRE_GPU and devices == ["cpu"]:
        raise RuntimeError(
            "Nenhuma GPU encontrada. Verifique a instalação do PyTorch com CUDA:\n"
            "  python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall\n"
            "Para rodar em CPU mesmo assim, defina REQUIRE_GPU = False no topo do script."
        )
    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    tasks     = [(v, devices[i % len(devices)]) for i, v in enumerate(variables)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # com múltiplas GPUs: processa variáveis em paralelo (uma por GPU)
    if len(devices) > 1 and len(variables) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(len(variables), len(devices))) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [train_variable(v, d) for v, d in tasks]

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
