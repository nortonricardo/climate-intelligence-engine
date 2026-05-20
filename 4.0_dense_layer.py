"""
4.0 — Rede Neural Densa (MLP) para gap-filling com busca de hiperparâmetros.

Arquitetura:
    Input(79) → [Linear → BN → GELU → Dropout] × N → Linear(1)
    Definida por Config.hidden_dims

Treinamento:
    Otimizador  : AdamW (lr=Config.lr, weight_decay=Config.weight_decay)
    Loss        : Huber (delta=0.05, robusto a outliers no espaço [0,1])
    Scheduler   : ReduceLROnPlateau (fator=0.5, patience=5)
    Early stop  : patience=25 épocas sem melhora no MAE de validação
    Val split   : últimos 10% do treino (temporal — sem leakage)
    Precisão    : float32 + AMP (mixed precision) quando GPU disponível
    Acumulação  : Config.accum_steps → batch_efetivo = BATCH_SIZE × accum_steps

GPU:
    Fila de tarefas: 15 tarefas (3 configs × 5 variáveis) distribuídas entre
    todas as GPUs disponíveis. Cada GPU consome a próxima tarefa disponível ao
    terminar a atual — sem desperdício. Sem GPU: fallback para CPU sequencial.

Input:
    data/{variable}_train_scaled.parquet
    data/{variable}_test_scaled.parquet

Output:
    models/{variable}_{config}_dense.pt
    results/4.0_dense_layer/{config}/{variable}/training_log.csv
    results/4.0_dense_layer/{config}/{variable}/metrics.csv
    results/4.0_dense_layer/{config}/metrics.csv
    results/4.0_dense_layer/comparison.csv   — comparação entre configs

Usage:
    python 4.0_dense_layer.py
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
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
torch.set_float32_matmul_precision("high")   # TF32 nos Tensor Cores da Ampere

# ── hiperparâmetros fixos ─────────────────────────────────────────────────────

BATCH_SIZE  = 131_072
MAX_EPOCHS  = 750
PATIENCE    = 25
VAL_RATIO   = 0.10
HUBER_DELTA = 0.05

VARIABLES_TO_RUN = None   # None = todas; ou ex: ["temperature"]
RESULTS_DIR      = _BASE_RESULTS / "4.0_dense_layer"
FEATURE_COLS     = get_feature_cols(k=15)
N_FEATURES       = len(FEATURE_COLS)   # 79


# ── configurações a comparar (uma por GPU) ────────────────────────────────────

@dataclass
class Config:
    name: str
    hidden_dims: list[int]
    dropout: float
    lr: float
    batch_size: int     = BATCH_SIZE
    weight_decay: float = 1e-4
    accum_steps: int    = 1    # passos de acumulação de gradiente

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.accum_steps


CONFIGS: list[Config] = [
    # base — modelo pequeno, batch grande para saturar a GPU (327K params)
    Config(
        name="base",
        hidden_dims=[256, 512, 256, 128, 64],
        dropout=0.20,
        lr=4e-3,
        batch_size=524_288,
        accum_steps=1,
    ),
    # wide — ~3GB de ativações, batch 2× (9M params)
    Config(
        name="wide",
        hidden_dims=[1024, 2048, 2048, 1024, 512],
        dropout=0.15,
        lr=3e-3,
        batch_size=262_144,
        accum_steps=2,
    ),
    # xl — ~5GB de ativações, batch padrão (19M params)
    Config(
        name="xl",
        hidden_dims=[2048, 4096, 2048, 1024, 512],
        dropout=0.10,
        lr=2e-3,
        batch_size=131_072,
        accum_steps=4,
    ),
]


# ── GPU detection ─────────────────────────────────────────────────────────────

def available_devices() -> list[str]:
    if torch.cuda.is_available():
        devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        print(f"  GPUs detectadas: {len(devs)}")
        for d in devs:
            name = torch.cuda.get_device_name(d)
            mem  = torch.cuda.get_device_properties(d).total_memory / 1024**3
            print(f"    {d}  {name}  ({mem:.1f} GB)")
        return devs
    print("  Nenhuma GPU detectada — usando CPU")
    return ["cpu"]


# ── arquitetura ───────────────────────────────────────────────────────────────

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


# ── dataset helpers ───────────────────────────────────────────────────────────

def df_to_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(df[FEATURE_COLS].to_numpy(dtype=np.float32).copy())
    y = torch.from_numpy(df["measurement"].to_numpy(dtype=np.float32).copy())
    return X, y


# ── treinamento ───────────────────────────────────────────────────────────────

def train_variable(variable: str, device_str: str, cfg: Config) -> dict | None:
    device  = torch.device(device_str)
    use_amp = device.type == "cuda"
    tag     = f"{variable}[{cfg.name}]"

    def _step(msg: str) -> None:
        print(f"  [{tag:<30s} {time.time() - t0:6.1f}s] {msg}", flush=True)

    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"\n[{tag}] → {device_str}", flush=True)

    # ── carrega dados ────────────────────────────────────────────────────────
    try:
        train_df = load_train(variable)
        test_df  = load_test(variable)
    except FileNotFoundError as e:
        print(f"  SKIP {variable}: {Path(e.filename).name} não encontrado — rode 1.6.")
        return None

    n_val    = max(1, int(len(train_df) * VAL_RATIO))
    val_df   = train_df.iloc[-n_val:].copy()
    train_df = train_df.iloc[:-n_val].copy()

    _step(f"treino={len(train_df):,}  val={len(val_df):,}  teste={len(test_df):,}")

    X_train, y_train = df_to_tensors(train_df); del train_df
    X_val,   y_val   = df_to_tensors(val_df);   del val_df

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
        drop_last=True,
    )

    # ── modelo ────────────────────────────────────────────────────────────────
    model = torch.compile(
        DenseNet(N_FEATURES, cfg.hidden_dims, cfg.dropout).to(device),
        mode="reduce-overhead",
    )
    criterion  = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if use_amp:
        mem_mb = torch.cuda.memory_allocated(device) / 1024**2
        mem_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        _step(
            f"modelo: {n_params:,} params"
            f"  |  batch_efetivo={cfg.effective_batch:,} (×{cfg.accum_steps})"
            f"  |  GPU: {mem_mb:.1f} MB / {mem_gb:.1f} GB"
        )

    # ── loop de épocas ────────────────────────────────────────────────────────
    var_dir   = RESULTS_DIR / cfg.name / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"{variable}_{cfg.name}_dense.pt"
    MODELS_DIR.mkdir(exist_ok=True)

    best_val_mae  = float("inf")
    no_improve    = 0
    log_rows: list[dict] = []
    epoch_times: list[float] = []

    X_val_dev = X_val.to(device)
    y_val_np  = y_val.numpy()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # treino com acumulação de gradiente
        model.train()
        epoch_loss = torch.zeros(1, device=device)
        optimizer.zero_grad(set_to_none=True)
        for i, (X_b, y_b) in enumerate(train_loader):
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(X_b)
                loss = criterion(pred, y_b) / cfg.accum_steps
            scaler_amp.scale(loss).backward()
            epoch_loss += loss.detach() * cfg.accum_steps
            if (i + 1) % cfg.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)
        train_loss = epoch_loss.item() / len(train_loader)  # sync único por época

        # validação em batches
        model.eval()
        val_chunks: list[np.ndarray] = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for vs in range(0, len(X_val_dev), cfg.batch_size * 4):
                val_chunks.append(
                    model(X_val_dev[vs : vs + cfg.batch_size * 4]).cpu().numpy()
                )
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

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve   = 0
            state = getattr(model, "_orig_mod", model).state_dict()
            torch.save(state, ckpt_path)
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

    pd.DataFrame(log_rows).to_csv(var_dir / "training_log.csv", index=False)

    # ── avalia no teste com o melhor modelo ───────────────────────────────────
    _step("avaliando no teste com o melhor modelo...")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()

    X_test, y_test = df_to_tensors(test_df); del test_df
    preds: list[np.ndarray] = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for s in range(0, len(X_test), cfg.batch_size * 4):
            chunk = X_test[s : s + cfg.batch_size * 4].to(device, non_blocking=True)
            preds.append(model(chunk).cpu().numpy())
    y_pred    = np.concatenate(preds)
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
        metrics, RESULTS_DIR / cfg.name, variable,
        extra_cols={
            "config": cfg.name,
            "n_train": n_train_final,
            "n_test": n_test,
            "best_val_mae": round(best_val_mae, 6),
        },
    )
    _step(f"→ {ckpt_path.name} + {out.name}  ({time.time() - t0:.0f}s total)")

    return {
        "config": cfg.name, "variable": variable,
        "n_train": n_train_final, "n_test": n_test,
        **metrics,
    }


# ── worker: uma GPU consome tarefas da fila até esgotar ──────────────────────

def _gpu_worker(device_str: str, task_q, result_q) -> None:
    torch.set_num_threads(8)
    while True:
        item = task_q.get()
        if item is None:          # poison pill — encerra o worker
            break
        cfg, variable = item
        result_q.put(train_variable(variable, device_str, cfg))


# ── main ──────────────────────────────────────────────────────────────────────

REQUIRE_GPU = True   # False → permite fallback para CPU


def main() -> None:
    print("=== 4.0 Dense Layer (MLP) ===\n")
    print(f"  max_epochs={MAX_EPOCHS}  patience={PATIENCE}\n")
    for cfg in CONFIGS:
        print(
            f"  [{cfg.name}]  arch={cfg.hidden_dims}  dropout={cfg.dropout}"
            f"  lr={cfg.lr}  batch={cfg.batch_size:,}"
            f"  accum={cfg.accum_steps}  efetivo={cfg.effective_batch:,}"
        )
    print()

    devices = available_devices()
    if REQUIRE_GPU and devices == ["cpu"]:
        raise RuntimeError(
            "Nenhuma GPU encontrada. Verifique a instalação do PyTorch com CUDA:\n"
            "  python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall\n"
            "Para rodar em CPU mesmo assim, defina REQUIRE_GPU = False no topo do script."
        )

    variables = VARIABLES_TO_RUN if VARIABLES_TO_RUN is not None else VARIABLES
    tasks     = [(cfg, v) for v in variables for cfg in CONFIGS]

    print(f"  {len(tasks)} tarefas  ({len(CONFIGS)} configs × {len(variables)} variáveis)"
          f"  →  {len(devices)} GPUs\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(devices) == 1:
        # single GPU: roda direto sem multiprocessing
        rows_all = []
        for cfg, v in tasks:
            r = train_variable(v, devices[0], cfg)
            if r is not None:
                rows_all.append(r)
    else:
        # múltiplas GPUs: fila de tarefas — cada GPU consome até esgotar
        ctx      = mp.get_context("spawn")
        task_q   = ctx.Queue()
        result_q = ctx.Queue()

        for task in tasks:
            task_q.put(task)
        for _ in devices:          # poison pill por worker
            task_q.put(None)

        procs = [
            ctx.Process(target=_gpu_worker, args=(dev, task_q, result_q))
            for dev in devices
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        rows_all = [result_q.get() for _ in tasks]
        rows_all = [r for r in rows_all if r is not None]

    if not rows_all:
        print("\nNenhuma variável processada.")
        return

    # ── resumo e comparação entre configs ─────────────────────────────────────
    df = pd.DataFrame(rows_all).round(4)

    print(f"\n=== Resumo por config ===")
    for cfg_name, grp in df.groupby("config"):
        print(f"\n  [{cfg_name}]")
        print(grp.set_index("variable")[["MAE", "RMSE", "R²", "r"]].to_string())

    if df["config"].nunique() > 1:
        comparison = df.pivot_table(
            index="variable", columns="config", values=["MAE", "RMSE", "R²"]
        ).round(4)
        comparison.to_csv(RESULTS_DIR / "comparison.csv")
        print(f"\n=== Comparação (MAE) ===")
        print(comparison["MAE"].to_string())
        print(f"\n→ {RESULTS_DIR}/comparison.csv")
    else:
        df.set_index("variable").to_csv(RESULTS_DIR / "metrics.csv")
        print(f"\n→ {RESULTS_DIR}/metrics.csv")


if __name__ == "__main__":
    main()
