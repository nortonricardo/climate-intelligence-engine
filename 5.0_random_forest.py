"""
5.0_random_forest.py — Random Forest via LightGBM RF mode (GPU).

Modo orquestrador:  python 5.0_random_forest.py
Modo worker:        python 5.0_random_forest.py --variable <var> --gpu <id>

Paralelismo: CUDA_VISIBLE_DEVICES + subprocess.Popen (uma variável por GPU).
Sem multiprocessing.spawn para evitar os problemas observados no 4.0.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# garante que utils.py seja encontrado independente de onde o script é chamado
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    MODELS_DIR,
    RESULTS_DIR,
    VARIABLES,
    compute_metrics,
    get_feature_cols,
    load_test,
    load_train,
    save_metrics,
)

# pasta de saída específica deste script
RESULTS_5 = RESULTS_DIR / "5.0_random_forest"

# ── hiperparâmetros ────────────────────────────────────────────────────────────

_BASE_PARAMS: dict = {
    # LightGBM RF mode: cada árvore é independente (sem boosting sequencial).
    # Equivale a um Random Forest clássico, mas usando histogramas de gradiente
    # acelerados por GPU — muito mais rápido que sklearn para 47M amostras.
    "boosting_type":     "rf",

    # Obrigatório para RF mode: cada árvore contribui com 100% do seu valor.
    # Sem isso, o padrão (0.1) aplica shrinkage como em boosting — o modelo
    # nunca converge e para de melhorar após 2-10 árvores.
    "learning_rate":     1.0,

    # Minimiza o MAE diretamente (mais robusto a outliers do que MSE).
    # No contexto de gap-filling climático, picos extremos de chuva/radiação
    # não distorcem o treinamento.
    "objective":         "regression_l1",
    "metric":            "mae",

    # Número máximo de folhas por árvore.
    # 255 = 2^8 - 1: árvores mais profundas capturam interações complexas entre
    # vizinhos (ex: n01 alto + n03 baixo + altitude delta grande). Com 47M amostras
    # e min_child_samples=20, overfitting é mínimo.
    "num_leaves":        255,
    "max_depth":         -1,

    # Exige pelo menos 20 amostras por folha terminal.
    # Com ~47M linhas, 20 amostras por folha é estatisticamente robusto e permite
    # que o modelo capture padrões raros sem overfitting.
    "min_child_samples": 20,

    # feature_fraction: 33% das features por árvore (nível de árvore).
    # feature_fraction_bynode: 33% das features em cada split individual.
    # Usar os dois juntos replica o comportamento do sklearn RF — subsampling
    # de features em CADA nó de decisão, não só uma vez por árvore.
    # Isso aumenta a diversidade entre árvores e reduz correlação entre elas.
    "feature_fraction":        1.0,
    "feature_fraction_bynode": 0.33,

    # A cada árvore, usa 80% das linhas sorteadas sem reposição (bagging).
    # bagging_freq=1 significa que o sorteio ocorre em toda árvore — obrigatório
    # para ativar o RF mode no LightGBM; sem isso boosting_type="rf" é ignorado.
    "bagging_fraction":  0.80,
    "bagging_freq":      1,

    # Regularização L2 nos valores das folhas. Penaliza pesos grandes,
    # reduz overfitting sem custo computacional relevante.
    "reg_lambda":        0.1,

    # Usa GPU para construção dos histogramas (operação mais custosa do treino).
    # gpu_use_dp=False força float32 — mais rápido que float64 e suficiente
    # para a precisão exigida pelo problema.
    "device":            "gpu",
    "gpu_use_dp":        False,

    # Threads de CPU por processo. Com 5 subprocessos em paralelo em 48 threads:
    # 5 × 8 = 40 threads, deixando 8 livres para o OS e demais processos.
    "num_threads":       8,

    # Suprime o log interno do LightGBM; o progresso é controlado pelo
    # callback lgb.log_evaluation(50) abaixo.
    "verbose":           -1,
}

# Número máximo de árvores a treinar por variável.
# 1000 dá runway suficiente para variáveis que precisam de mais árvores para
# convergir (ex: rainfall, global_radiation). Early stopping garante que para
# antes se não houver melhora.
N_ESTIMATORS = 1000

# Para o treino se o val MAE não melhorar em 50 árvores consecutivas.
# Para RF, a melhora desacelera conforme as árvores se acumulam — 50 dá margem
# suficiente para confirmar convergência antes de parar.
EARLY_STOP   = 50

# Fração do treino reservada para validação (split temporal — as últimas linhas).
# Mantém consistência com os splits usados em 3.0 e 4.0.
VAL_FRACTION = 0.10


# ── worker (executado no subprocess) ──────────────────────────────────────────

def _worker(variable: str, gpu_id: int) -> None:
    """Treina um RF para uma única variável. Chamado pelo subprocess filho."""

    # lista das 79 features usadas pelo pipeline (n01..n15, d01..d15, etc.)
    feat_cols = get_feature_cols(k=15)
    label_col = "measurement"
    cols      = feat_cols + [label_col]

    # carrega apenas as colunas necessárias para economizar RAM
    print(f"[{variable}] carregando treino...", flush=True)
    df_train = load_train(variable, columns=cols)

    # split temporal: os últimos VAL_FRACTION% do treino viram validação.
    # Não embaralha — respeita a ordem cronológica dos dados.
    n_val = int(len(df_train) * VAL_FRACTION)
    n_tr  = len(df_train) - n_val

    # converte para float32 contíguo em memória (LightGBM exige array C-contiguous)
    X_tr  = df_train.iloc[:n_tr][feat_cols].values.astype(np.float32)
    y_tr  = df_train.iloc[:n_tr][label_col].values.astype(np.float32)
    X_val = df_train.iloc[n_tr:][feat_cols].values.astype(np.float32)
    y_val = df_train.iloc[n_tr:][label_col].values.astype(np.float32)

    # libera o DataFrame — os arrays numpy acima já têm cópia dos dados
    del df_train

    print(f"[{variable}] treino={n_tr:,}  val={n_val:,}", flush=True)

    # lgb.Dataset é lazy: só constrói a representação interna de histogramas
    # quando lgb.train() é chamado. free_raw_data=True diz ao LightGBM para
    # liberar as referências aos arrays numpy assim que os histogramas estiverem
    # prontos, recuperando ~14GB de RAM durante o treino.
    ds_train = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=True)
    ds_val   = lgb.Dataset(
        X_val, label=y_val, feature_name=feat_cols,
        free_raw_data=True,
        reference=ds_train,  # garante que val usa os mesmos bins do treino
    )

    # remove as referências Python; free_raw_data=True cuida do resto durante lgb.train()
    del X_tr, y_tr, X_val, y_val

    # injeta o ID da GPU visível neste subprocess (sempre 0 porque CUDA_VISIBLE_DEVICES
    # já remapeou a GPU física para índice 0 antes de o processo ser iniciado)
    params = {**_BASE_PARAMS, "gpu_device_id": gpu_id}

    print(f"[{variable}] iniciando RF — GPU {gpu_id}", flush=True)
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[
            lgb.log_evaluation(50),               # imprime val MAE a cada 50 árvores
            lgb.early_stopping(EARLY_STOP, verbose=True),  # para se não melhorar em 50
        ],
    )

    # salva o modelo no formato nativo do LightGBM (.lgb).
    # Pode ser recarregado com lgb.Booster(model_file="...lgb") para inferência.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{variable}_rf.lgb"
    model.save_model(str(model_path))
    print(f"[{variable}] modelo salvo → {model_path}", flush=True)

    # ── feature importance ────────────────────────────────────────────────────
    # gain: soma do ganho de MAE proporcionado por cada feature em todos os splits.
    #       Mais informativo — features com alto gain contribuem mais para a precisão.
    # split: número de vezes que a feature foi usada em um split.
    #        Pode inflar features com muitos valores únicos (ex: distância contínua).
    var_dir = RESULTS_5 / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    fi_df = pd.DataFrame({
        "feature": feat_cols,
        "gain":    model.feature_importance(importance_type="gain"),
        "split":   model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi_df.to_csv(var_dir / "feature_importance.csv", index=False)

    # ── métricas no conjunto de teste ─────────────────────────────────────────
    print(f"[{variable}] avaliando no teste...", flush=True)
    df_test = load_test(variable, columns=cols)
    X_test  = df_test[feat_cols].values.astype(np.float32)
    y_test  = df_test[label_col].values.astype(np.float32)
    del df_test

    # usa apenas as árvores até best_iteration (ponto de menor val MAE)
    y_pred  = model.predict(X_test, num_iteration=model.best_iteration)

    # calcula MAE, RMSE, R², Bias e r de Pearson no espaço MinMax [0,1]
    metrics = compute_metrics(y_test, y_pred)

    # salva em results/5.0_random_forest/{variable}/metrics.csv
    save_metrics(
        metrics,
        RESULTS_5,
        variable,
        extra_cols={
            "n_train":       n_tr,
            "n_estimators":  model.best_iteration,  # árvores efetivas (≤ N_ESTIMATORS)
        },
    )
    print(
        f"[{variable}] MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}"
        f"  R²={metrics['R²']:.4f}  r={metrics['r']:.4f}",
        flush=True,
    )


# ── orquestrador ──────────────────────────────────────────────────────────────

def _aggregate_summary() -> None:
    """Lê os metrics.csv de cada variável e consolida em um único resumo."""
    rows = []
    for var in VARIABLES:
        p = RESULTS_5 / var / "metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "variable", var)
            rows.append(df)
    if not rows:
        return
    summary = pd.concat(rows, ignore_index=True)
    out = RESULTS_5 / "metrics.csv"
    summary.to_csv(out, index=False)
    print(f"\nResumo → {out}")
    print(summary.to_string(index=False))


def main() -> None:
    """
    Orquestrador: detecta GPUs e lança um subprocess por variável.

    Cada subprocess recebe CUDA_VISIBLE_DEVICES=X, tornando somente a GPU X
    visível. Dentro do subprocess, a GPU aparece como device 0, então --gpu 0
    é sempre passado. Isso evita conflitos de alocação entre processos e elimina
    a necessidade de multiprocessing.spawn (que causou crashes no 4.0).
    """
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    if n_gpus == 0:
        print("Nenhuma GPU detectada — LightGBM usará CPU.", flush=True)

    RESULTS_5.mkdir(parents=True, exist_ok=True)
    print(f"GPUs detectadas: {n_gpus}")

    # lança todos os subprocessos imediatamente (não espera um terminar para começar o próximo)
    procs: list[tuple[str, subprocess.Popen]] = []
    for i, var in enumerate(VARIABLES):
        # distribui variáveis entre GPUs em round-robin (5 vars, 5 GPUs → 1:1)
        gpu_id = i % max(n_gpus, 1)

        # copia o ambiente atual e sobrescreve apenas CUDA_VISIBLE_DEVICES
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

        # chama o próprio script com --variable para entrar no modo worker
        cmd = [sys.executable, __file__, "--variable", var, "--gpu", "0"]
        print(f"  → {var:20s}  GPU {gpu_id}")
        p = subprocess.Popen(cmd, env=env)
        procs.append((var, p))

    # aguarda cada processo terminar e coleta o exit code
    failed = []
    for var, p in procs:
        ret    = p.wait()
        status = "OK  " if ret == 0 else f"ERRO (exit={ret})"
        print(f"  [{status}] {var}", flush=True)
        if ret != 0:
            failed.append(var)

    if failed:
        print(f"\nVariáveis com falha: {failed}", flush=True)

    # consolida metrics.csv de todas as variáveis em um resumo único
    _aggregate_summary()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variable", default=None,  help="variável a treinar (modo worker)")
    parser.add_argument("--gpu",      type=int, default=0, help="GPU device ID visível no subprocess")
    args = parser.parse_args()

    # sem --variable → orquestrador; com --variable → worker
    if args.variable is not None:
        _worker(args.variable, args.gpu)
    else:
        main()
