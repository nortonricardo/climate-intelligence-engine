"""
6.0_transformer.py — SNT (Spatial Neighbor Transformer) com DDP multi-GPU.

Arquitetura SNT:
    Cada um dos 15 vizinhos vira um token geoespacial: Linear([n, d, a, b_sin, b_cos] → D).
    O contexto temporal vira um token separado:         Linear([h_sin, h_cos, doy_sin, doy_cos] → D).
    16 tokens → TransformerEncoder (pre-norm) → mean pooling → MLP head → predição.
    Vizinhos com positional encoding aprendível + schedule warmup linear → cosine decay.

GPU:
    Variável por variável, cada config usa TODAS as GPUs via DDP (DistributedDataParallel).
    Gradientes são sincronizados via NCCL após cada batch → treino 5× mais rápido que 1 GPU.
    Ordem: temperature/base → temperature/wide → temperature/xl → humidity/base → ...

Modo orquestrador: python 6.0_transformer.py
Modo DDP worker:   chamado internamente via torchrun (não invocar manualmente)

Input:  data/{variable}_train_scaled.parquet  /  data/{variable}_test_scaled.parquet
Output: models/{variable}_{config}_snt.pt
        results/6.0_transformer/{config}/{variable}/training_log.csv
        results/6.0_transformer/{config}/{variable}/metrics.csv
        results/6.0_transformer/comparison.csv
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist           # primitivas de comunicação entre GPUs (all_reduce, broadcast)
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP  # wrapper que sincroniza gradientes
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

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

# cudnn.benchmark=True faz o PyTorch testar múltiplos algoritmos de convolução no primeiro
# batch e escolher o mais rápido para o tamanho de entrada atual — ganho de ~5-10% no Transformer.
torch.backends.cudnn.benchmark = True

# TF32 usa aritmética de 19 bits nos Tensor Cores das GPUs Ampere (A100, RTX 30xx).
# Mantém a precisão de float32 para a maior parte dos cálculos mas executa 8× mais rápido
# nas multiplicações de matrizes — principal operação do Transformer (Q·K^T).
torch.set_float32_matmul_precision("high")

# Força o backend "math" do SDPA dispatcher do PyTorch (desabilita Flash e memory-efficient).
#
# Por que não Flash Attention: falha com "CUDA error: invalid argument" nesta combinação
# de driver CUDA + head_dim=16 (limite do FA2 nesta versão).
#
# Por que não memory-efficient: falha com "batch size exceeds 65535" — nosso batch base
# é 262.144 por GPU, acima do limite interno do kernel xformers.
#
# O backend math não tem restrição de batch nem de head_dim.
# Para sequências curtas (16 tokens), a matriz de atenção é 16×16 — tão pequena que
# o math é comparável ao Flash em velocidade (Flash só ganha em seq_len >> 512).
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

RESULTS_6    = RESULTS_DIR / "6.0_transformer"
K            = 15                    # número de vizinhos — deve bater com get_feature_cols(k=15)
FEATURE_COLS = get_feature_cols(k=K) # lista das 79 colunas de features na ordem correta
N_FEATURES   = len(FEATURE_COLS)     # 79

MAX_EPOCHS   = 750   # teto de épocas; early stopping normalmente para antes
PATIENCE     = 25    # épocas consecutivas sem melhora no val MAE para parar
VAL_FRACTION = 0.10  # últimos 10% do treino como validação (split temporal)
HUBER_DELTA   = 0.05  # ponto de transição MAE↔MSE da Huber Loss no espaço [0,1]
WARMUP_EPOCHS = max(5, MAX_EPOCHS // 20)  # warmup linear de 37 épocas (~5% de MAX_EPOCHS)


# ── configurações ──────────────────────────────────────────────────────────────
#
# Três tamanhos de modelo para cobrir a curva capacidade × tempo:
#   base  → referência rápida, detecta bugs e valida pipeline
#   wide  → melhor custo-benefício (baseado em 4.0 onde wide ≈ xl)
#   xl    → teto de capacidade, lr menor para estabilidade
#
# Regra obrigatória: embedding_dim % n_heads == 0.
# head_dim = embedding_dim // n_heads — dimensão por cabeça de atenção.
# Valores típicos: head_dim ∈ {16, 32, 64}.
#
# Com DDP e 5 GPUs, batch efetivo = batch_size × 5:
#   base:  262.144 × 5 = 1.310.720 amostras por step
#   wide:  131.072 × 5 =   655.360 amostras por step
#   xl:     65.536 × 5 =   327.680 amostras por step

CONFIGS: dict[str, dict] = {
    "base": {
        "embedding_dim": 64,    # D — dimensão de cada token no espaço de embedding
        "n_layers":      3,     # número de blocos Transformer empilhados
        "n_heads":       4,     # cabeças de atenção (head_dim = 64/4 = 16)
        "ffn_hidden":    256,   # dimensão da camada oculta do FFN = 4 × embedding_dim
        "dropout":       0.10,  # dropout aplicado dentro da atenção e do FFN
        "batch_size":    262_144,
        "lr":            1e-3,
    },
    "wide": {
        "embedding_dim": 128,
        "n_layers":      6,
        "n_heads":       8,     # head_dim = 128/8 = 16
        "ffn_hidden":    512,
        "dropout":       0.10,
        "batch_size":    131_072,
        "lr":            1e-3,
    },
    "xl": {
        "embedding_dim": 256,
        "n_layers":      12,
        "n_heads":       8,     # head_dim = 256/8 = 32
        "ffn_hidden":    1024,
        "dropout":       0.10,
        "batch_size":    65_536,
        "lr":            5e-4,  # modelo maior é mais sensível → lr menor para estabilidade
    },
}


# ── modelo ─────────────────────────────────────────────────────────────────────

class SNT(nn.Module):
    """
    Spatial Neighbor Transformer.

    Motivação: os 79 features têm estrutura geoespacial natural —
    15 vizinhos × 5 atributos cada. Tratar cada vizinho como um token
    permite que a atenção aprenda explicitamente "quais estações vizinhas
    importam mais" para cada predição, em vez de deixar isso implícito
    como no MLP.

    Estrutura dos 79 features de entrada (get_feature_cols k=15):
        n01..n15      índices  0..14   medição dos K vizinhos         [0,1]
        d01..d15      índices 15..29   distância normalizada           [0,1]
        a01..a15      índices 30..44   delta altitude normalizado      [0,1]
        b01..b15_sin  índices 45..59   azimute seno                  [-1,1]
        b01..b15_cos  índices 60..74   azimute cosseno               [-1,1]
        hour_sin/cos  índices 75..76   hora do dia cíclica           [-1,1]
        doy_sin/cos   índices 77..78   dia do ano cíclico            [-1,1]

    Tokens gerados:
        15 neighbor tokens — Linear(5 → D) + positional encoding aprendível por rank
         1 temporal token  — Linear(4 → D)
        Total: 16 tokens × D → self-attention 16×16 por cabeça.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_layers:      int,
        n_heads:       int,
        ffn_hidden:    int,
        dropout:       float,
    ) -> None:
        super().__init__()

        # Dois embeddings distintos: neighbor (5 features geoespaciais) e temporal (4 features).
        # Usar embeddings separados é importante porque os dois grupos têm semântica diferente:
        #   neighbor: [medição, distância, altitude, azimute_sin, azimute_cos] — descreve
        #             a relação física entre a estação-alvo e cada vizinho.
        #   temporal: [hour_sin, hour_cos, doy_sin, doy_cos] — descreve o contexto de tempo.
        # Se usássemos um único Linear(5→D) com padding=0 no temporal, o modelo poderia
        # confundir o token temporal (d=0, a=0) com um vizinho co-localizado — o que não existe.
        self.neighbor_embed = nn.Linear(5, embedding_dim)
        self.temporal_embed = nn.Linear(4, embedding_dim)

        # Embedding de posição aprendível para os K vizinhos (ranqueados por distância crescente).
        # Complementa d01..d15 (distância contínua) com um bias por rank ordinal: "ser o 1º
        # vizinho" é estruturalmente diferente de "ser o 15º" — o modelo aprende esse padrão.
        # Custo: K × D parâmetros extras (ex: 15×64 = 960 no base config).
        self.neighbor_pos = nn.Embedding(K, embedding_dim)

        # LayerNorm após concatenar os 16 tokens embedados.
        # Antes de entrar no Transformer, normaliza a escala dos embeddings para
        # que os dois tipos de tokens (neighbor e temporal) fiquem na mesma faixa de valores.
        self.embed_norm = nn.LayerNorm(embedding_dim)

        # Transformer Encoder: N blocos idênticos de [MultiHeadAttention + FFN].
        #
        # norm_first=True → pre-norm: LayerNorm é aplicado ANTES da atenção e do FFN.
        # O post-norm (padrão original do paper "Attention Is All You Need") aplica
        # LayerNorm depois das residual connections. Com modelos profundos (≥6 camadas),
        # pre-norm converge mais facilmente porque o gradiente flui pelas skip connections
        # sem passar pelo LayerNorm — evitando vanishing gradients.
        #
        # activation="gelu": GELU tem comportamento suave em zero (não é um degrau como ReLU),
        # o que melhora o fluxo de gradientes no FFN.
        #
        # batch_first=True: convenção (batch, seq, dim) em vez de (seq, batch, dim) do PyTorch
        # original — mais legível e compatível com nosso reshape dos tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            # enable_nested_tensor=False: desativa uma otimização interna do PyTorch que
            # pode causar warnings quando batch_first=True e não há padding na sequência.
            # Nossa sequência tem sempre 16 tokens completos — sem padding.
            enable_nested_tensor=False,
        )

        # MLP Head: converte o vetor agregado (mean pooling) em uma predição escalar.
        #   LayerNorm → estabiliza a saída do Transformer antes da projeção linear.
        #   Linear(D → D//2) → comprime a representação.
        #   GELU → não-linearidade suave.
        #   Dropout → regularização final.
        #   Linear(D//2 → 1) → predição do valor da medição na estação-alvo.
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

        # Tensor de índices [0, 1, ..., K-1] pré-computado e registrado como buffer.
        # register_buffer: o tensor é movido automaticamente para o device correto
        # (GPU) junto com o modelo, sem ser tratado como parâmetro treinável.
        # Sem isso, precisaríamos criar torch.arange(K) a cada forward pass —
        # pequena mas evitável alocação de memória.
        self.register_buffer("_idx", torch.arange(K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 79) — features na ordem de get_feature_cols(k=15)
        →   (batch,)    — predição escalar em [0, 1]
        """
        idx = self._idx  # (K,) — tensor de índices já no device correto

        # ── constrói os 15 neighbor tokens ────────────────────────────────────
        # Para o vizinho i, agrupa suas 5 features em um único vetor:
        #   x[:, idx]        → (batch, K): medições n01..n15
        #   x[:, K+idx]      → (batch, K): distâncias d01..d15
        #   x[:, 2K+idx]     → (batch, K): altitudes a01..a15
        #   x[:, 3K+idx]     → (batch, K): azimutes sin b01_sin..b15_sin
        #   x[:, 4K+idx]     → (batch, K): azimutes cos b01_cos..b15_cos
        #
        # torch.stack([t1,t2,t3,t4,t5], dim=2): cada ti tem forma (batch, K);
        # empilhar em dim=2 cria (batch, K, 5) — K tokens, cada um com 5 features.
        neighbor_tokens = torch.stack([
            x[:, idx],
            x[:, K   + idx],
            x[:, 2*K + idx],
            x[:, 3*K + idx],
            x[:, 4*K + idx],
        ], dim=2)  # (batch, K, 5)

        # ── constrói o temporal token ──────────────────────────────────────────
        # Últimas 4 features (índices 75..78): hour_sin, hour_cos, doy_sin, doy_cos.
        # unsqueeze(1) adiciona a dimensão de sequência → (batch, 1, 4).
        temporal_token = x[:, 5*K : 5*K + 4].unsqueeze(1)  # (batch, 1, 4)

        # ── embedding ─────────────────────────────────────────────────────────
        # Cada token é projetado para o espaço de dimensão D via Linear.
        # O modelo aprende quais combinações de [n, d, a, b_sin, b_cos] são relevantes.
        neighbor_emb = self.neighbor_embed(neighbor_tokens)  # (batch, K, D)
        # Positional encoding: soma bias aprendível ao embedding de cada vizinho.
        # idx = [0,..,K-1] já está no device correto (register_buffer).
        # (K, D) é adicionado a (batch, K, D) por broadcasting automático do PyTorch.
        neighbor_emb = neighbor_emb + self.neighbor_pos(idx)  # (batch, K, D)
        temporal_emb = self.temporal_embed(temporal_token)    # (batch, 1, D)

        # ── sequência de 16 tokens ─────────────────────────────────────────────
        # Concatena neighbor tokens (15) e temporal token (1) na dimensão de sequência.
        # LayerNorm normaliza os valores antes do Transformer.
        seq = torch.cat([neighbor_emb, temporal_emb], dim=1)  # (batch, 16, D)
        seq = self.embed_norm(seq)

        # ── self-attention ────────────────────────────────────────────────────
        # Cada token "olha" para todos os outros 15 tokens via atenção.
        # Para o wide config (H=8 cabeças): 8 matrizes de atenção 16×16 em paralelo.
        # Cada cabeça aprende um padrão diferente:
        #   ex: cabeça 1 = "vizinhos próximos"
        #       cabeça 2 = "mesma faixa de altitude"
        #       cabeça 3 = "mesmo azimute (norte/sul)"
        #       cabeça 4 = "contexto temporal → hora do pico solar"
        seq = self.transformer(seq)  # (batch, 16, D)

        # ── mean pooling ──────────────────────────────────────────────────────
        # Agrega os 16 tokens em um único vetor de contexto fazendo a média.
        # Alternativa seria usar um [CLS] token dedicado, mas mean pooling
        # tende a ser mais estável e não precisa de parâmetro extra.
        out = seq.mean(dim=1)  # (batch, D)

        # ── predição ──────────────────────────────────────────────────────────
        return self.head(out).squeeze(1)  # (batch,)


# ── DDP worker ─────────────────────────────────────────────────────────────────

def _ddp_worker(variable: str, config_name: str) -> None:
    """
    Executado em CADA processo lançado pelo torchrun.

    Com 5 GPUs, o torchrun inicia 5 cópias deste processo simultaneamente.
    Cada cópia recebe variáveis de ambiente injetadas pelo torchrun:
        RANK       = índice global do processo (0..4)
        WORLD_SIZE = total de processos (5)
        LOCAL_RANK = índice da GPU neste nó (0..4, igual ao RANK em 1 nó)

    Todos os 5 processos rodam o mesmo código, mas:
        - Cada um usa uma GPU diferente (local_rank 0..4)
        - Cada um processa 1/5 dos dados por época (DistributedSampler)
        - Os gradientes são somados entre todos após cada backward (all_reduce)
        - Apenas rank 0 imprime, salva modelos e avalia no teste
    """
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # init_process_group: inicializa o grupo de comunicação entre os processos.
    # backend="nccl": protocolo otimizado para comunicação GPU↔GPU via NVLink/PCIe.
    # O torchrun já configurou MASTER_ADDR e MASTER_PORT — o processo rank 0 age
    # como coordenador; os outros se conectam a ele na porta definida.
    # set_device ANTES de init_process_group: garante que o NCCL cria o comunicador
    # na GPU correta. Sem isso, o NCCL pode usar GPU 0 em todos os ranks até o primeiro
    # collective, e o barrier() emite warning "using device under current context".
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    cfg       = CONFIGS[config_name]
    label_col = "measurement"
    cols      = FEATURE_COLS + [label_col]

    # ── carrega dados ──────────────────────────────────────────────────────────
    # Cada rank carrega o dataset COMPLETO independentemente do parquet.
    # O DistributedSampler é responsável por dar a cada rank apenas 1/world_size
    # dos índices por época — a divisão de trabalho acontece no DataLoader.
    # Com 5 GPUs e 47M amostras: cada GPU processa ~9.4M amostras por época.
    if rank == 0:
        print(f"\n[{variable}/{config_name}] carregando treino...", flush=True)

    df_train = load_train(variable, columns=cols)
    n_val    = int(len(df_train) * VAL_FRACTION)
    n_tr     = len(df_train) - n_val

    # Split temporal: os últimos VAL_FRACTION% do treino viram validação.
    # Sem embaralhamento — preserva a ordem cronológica dos dados.
    X_tr  = df_train.iloc[:n_tr][FEATURE_COLS].values.astype(np.float32)
    y_tr  = df_train.iloc[:n_tr][label_col].values.astype(np.float32)
    X_val = df_train.iloc[n_tr:][FEATURE_COLS].values.astype(np.float32)
    y_val = df_train.iloc[n_tr:][label_col].values.astype(np.float32)
    del df_train  # libera o DataFrame — os arrays numpy acima têm cópia dos dados

    ds_train = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_val   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    del X_tr, y_tr, X_val, y_val

    # DistributedSampler divide os índices entre os ranks:
    #   rank 0 processa índices 0, 5, 10, 15, ...
    #   rank 1 processa índices 1, 6, 11, 16, ...
    #   ...
    # shuffle=True no treino: a ordem muda a cada época (via set_epoch).
    # shuffle=False na validação: ordem determinística para métricas reproduzíveis.
    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(ds_val,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        ds_train, batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=4,      # threads de CPU para pré-carregar batches em paralelo
        pin_memory=True,    # aloca batches em memória pinada (não paginável) → transferência GPU mais rápida
        drop_last=True,     # descarta o último batch incompleto — necessário para DDP evitar desbalanceamento
    )
    val_loader = DataLoader(
        ds_val, batch_size=cfg["batch_size"] * 2,  # validação sem gradientes → batch 2× maior cabe na GPU
        sampler=val_sampler, num_workers=4, pin_memory=True,
    )

    if rank == 0:
        eff_batch = cfg["batch_size"] * world_size
        print(
            f"[{variable}/{config_name}] treino={n_tr:,}  val={n_val:,}"
            f"  GPUs={world_size}  batch/GPU={cfg['batch_size']:,}"
            f"  batch_efetivo={eff_batch:,}",
            flush=True,
        )

    # ── modelo ────────────────────────────────────────────────────────────────
    model = SNT(
        embedding_dim=cfg["embedding_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ffn_hidden=cfg["ffn_hidden"],
        dropout=cfg["dropout"],
    ).to(device)

    # DDP envolve o modelo e intercepta o backward():
    #   1. Cada GPU calcula gradientes no seu shard de dados (forward + backward normal).
    #   2. Ao terminar o backward, DDP faz all_reduce dos gradientes entre todas as GPUs.
    #   3. all_reduce = soma os gradientes de todas as GPUs e divide por world_size.
    #   4. Resultado: todos os parâmetros recebem o mesmo gradiente médio.
    #   5. Cada GPU faz seu próprio optimizer.step() com o gradiente sincronizado.
    #   → Os parâmetros de todos os ranks ficam idênticos após cada step.
    #
    # find_unused_parameters=False: desativa a detecção de parâmetros não usados
    # no grafo de gradientes — mais eficiente pois todos os parâmetros do SNT
    # participam do forward (nenhum é condicional).
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # AdamW com weight_decay aplica regularização L2 nos pesos (mas não nos biases).
    # Transformers são sensíveis ao lr — começar com 1e-3 e deixar o scheduler reduzir.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=1e-4,
    )

    # Huber Loss: comportamento MSE para erros pequenos (|e| < delta) e MAE para grandes.
    # delta=0.05 em escala [0,1] significa que erros até 5% do range são tratados como MSE.
    # Mais robusta a outliers que MSE puro — importante para rainfall e global_radiation.
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    # Schedule em dois estágios:
    #
    #   1. Warmup linear (WARMUP_EPOCHS épocas): lr cresce de 1% → 100% do lr inicial.
    #      Transformers são instáveis nas primeiras épocas quando os embeddings ainda não
    #      têm direção útil e os gradientes têm alta variância. O warmup evita que um
    #      step grande no início destrua a inicialização dos pesos.
    #
    #   2. Cosine annealing (épocas restantes): lr decai suavemente seguindo
    #      lr(t) = eta_min + 0.5·(lr_max - eta_min)·(1 + cos(π·t/T)).
    #      O decaimento cosine é mais suave que reduções por plateau — o modelo
    #      "pousa" gradualmente no mínimo em vez de chegar em patamares discretos.
    #
    # scheduler.step() é chamado uma vez por época (não por batch).
    _warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS,
    )
    _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[_warmup, _cosine], milestones=[WARMUP_EPOCHS],
    )

    # GradScaler para mixed precision: mantém pesos em float32 mas faz o forward em
    # float16, reduzindo uso de memória e acelerando operações nos Tensor Cores.
    # O scaler ajusta automaticamente a escala para evitar underflow em float16.
    scaler = torch.amp.GradScaler("cuda")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{variable}/{config_name}] parâmetros: {n_params:,}", flush=True)

    # ── cria dirs de saída — apenas rank 0 faz I/O em disco ───────────────────
    var_dir   = RESULTS_6 / config_name / variable
    ckpt_path = MODELS_DIR / f"{variable}_{config_name}_snt.pt"
    if rank == 0:
        var_dir.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # dist.barrier(): todos os ranks esperam aqui até que rank 0 termine de criar
    # os diretórios. Sem isso, ranks 1..4 poderiam tentar acessar dirs inexistentes.
    dist.barrier()

    # ── loop de épocas ─────────────────────────────────────────────────────────
    best_val_mae   = float("inf")
    epochs_no_impr = 0
    log_rows: list[dict] = []
    epoch_times: list[float] = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # ── treino ────────────────────────────────────────────────────────────
        # set_epoch(epoch): atualiza a semente do DistributedSampler para que
        # cada época tenha uma ordem de shuffle diferente E igual em todos os ranks.
        # Sem isso, todos os ranks usariam a mesma ordem fixa → sem diversidade.
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = torch.zeros(1, device=device)

        for X_batch, y_batch in train_loader:
            # non_blocking=True: a transferência CPU→GPU acontece em paralelo com
            # outras operações, sem bloquear a CPU enquanto espera o dado chegar.
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # set_to_none=True: libera a memória dos gradientes em vez de zerá-los
            # (mais eficiente que zero_grad() padrão).
            optimizer.zero_grad(set_to_none=True)

            # autocast: executa o forward em float16 automaticamente nas ops que
            # se beneficiam (Linear, attention), mantendo float32 onde necessário
            # (BatchNorm, loss).
            with torch.amp.autocast("cuda"):
                pred = model(X_batch)
                loss = criterion(pred, y_batch)

            # scaler.scale(loss).backward(): escala a loss antes do backward para
            # evitar underflow dos gradientes em float16.
            scaler.scale(loss).backward()

            # unscale_ antes do clip: reverte a escala dos gradientes para que o
            # clipping opere na magnitude real (não na magnitude escalada).
            scaler.unscale_(optimizer)

            # Gradient clipping: limita a norma L2 dos gradientes a 1.0.
            # Transformers são suscetíveis a explosão de gradientes, especialmente
            # nas primeiras épocas. Clipping a 1.0 é o padrão recomendado.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)   # aplica o update (com gradientes reescalados)
            scaler.update()          # ajusta o fator de escala para o próximo step

            train_loss_sum += loss.detach()

        # all_reduce(SUM): soma as loss_sum de todos os ranks em todos os ranks.
        # Dividir por (batches × world_size) dá a loss média global da época.
        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
        train_loss_avg = (train_loss_sum / (len(train_loader) * world_size)).item()

        # ── validação ─────────────────────────────────────────────────────────
        model.eval()
        val_abs_sum = torch.zeros(1, device=device)  # soma de |pred - y| em todos os ranks
        val_n       = torch.zeros(1, device=device)  # contagem total de amostras

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                pred    = model(X_batch)
                val_abs_sum += torch.abs(pred - y_batch).sum()
                val_n       += y_batch.numel()

        # Agrega os valores parciais de todos os ranks para calcular o MAE global.
        # Após o all_reduce, val_abs_sum e val_n têm os totais de todas as GPUs.
        # val_mae = MAE calculado sobre o conjunto de validação completo.
        dist.all_reduce(val_abs_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_n,       op=dist.ReduceOp.SUM)
        val_mae = (val_abs_sum / val_n).item()

        scheduler.step()  # avança o schedule (warmup → cosine), independente do val_mae
        current_lr  = optimizer.param_groups[0]["lr"]
        epoch_dt    = time.time() - epoch_start
        epoch_times.append(epoch_dt)

        if rank == 0:
            log_rows.append({
                "epoch":      epoch,
                "train_loss": round(train_loss_avg, 6),
                "val_mae":    round(val_mae, 6),
                "lr":         current_lr,
            })
            # ETA estimado com base na média das últimas 10 épocas
            avg_dt  = sum(epoch_times[-10:]) / len(epoch_times[-10:])
            eta_s   = avg_dt * (MAX_EPOCHS - epoch)
            eta_str = (f"{int(eta_s // 3600)}h{int(eta_s % 3600 // 60):02d}m"
                       if eta_s >= 3600 else f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s")

            if epoch % 10 == 0 or val_mae < best_val_mae:
                mark = "  ✓" if val_mae < best_val_mae else f"  ({epochs_no_impr}/{PATIENCE})"
                print(
                    f"  [{variable}/{config_name}  {time.time()-t0:6.0f}s]"
                    f"  epoch {epoch:4d}/{MAX_EPOCHS}"
                    f"  loss={train_loss_avg:.5f}"
                    f"  val_mae={val_mae:.5f}"
                    f"  best={best_val_mae:.5f}"
                    f"  lr={current_lr:.2e}"
                    f"  Δt={epoch_dt:.1f}s  eta={eta_str}"
                    f"{mark}",
                    flush=True,
                )

        # ── early stopping + salva melhor modelo ──────────────────────────────
        # Problema: todos os ranks precisam tomar a mesma decisão de parar.
        # Se só rank 0 checar best_val_mae, os outros ranks podem continuar
        # por mais épocas → dessincronia → deadlock no próximo all_reduce.
        #
        # Solução: rank 0 calcula se houve melhora, converte em tensor (1.0 ou 0.0)
        # e faz broadcast para todos os ranks. Assim todos tomam a mesma decisão.
        improved = torch.tensor(1.0 if val_mae < best_val_mae else 0.0, device=device)
        dist.broadcast(improved, src=0)  # rank 0 → todos os outros ranks

        if improved.item() > 0.5:
            best_val_mae   = val_mae
            epochs_no_impr = 0
            if rank == 0:
                # model.module: o DDP é um wrapper — .module acessa o SNT interno.
                # Salvar model.module.state_dict() em vez de model.state_dict()
                # garante que o checkpoint não tem o prefixo "module." nos nomes
                # dos parâmetros, facilitando o carregamento sem DDP na inferência.
                torch.save(model.module.state_dict(), ckpt_path)
        else:
            epochs_no_impr += 1

        if epochs_no_impr >= PATIENCE:
            if rank == 0:
                print(
                    f"  [{variable}/{config_name}] early stop na época {epoch}"
                    f"  best_val_mae={best_val_mae:.5f}",
                    flush=True,
                )
            break

    # ── log de treino (rank 0) ─────────────────────────────────────────────────
    if rank == 0:
        pd.DataFrame(log_rows).to_csv(var_dir / "training_log.csv", index=False)

    # ── avalia no teste com o melhor modelo (rank 0) ───────────────────────────
    # Avaliação é single-GPU: rank 0 carrega o melhor checkpoint e prediz o teste.
    # Os outros ranks já terminaram sua participação útil; apenas rank 0 faz I/O.
    if rank == 0:
        print(f"  [{variable}/{config_name}] avaliando no teste...", flush=True)

        # Instancia um SNT limpo (sem DDP) e carrega o melhor state_dict.
        # weights_only=True: carrega apenas tensores, sem executar código arbitrário
        # do arquivo pickle — prática de segurança recomendada pelo PyTorch.
        eval_model = SNT(
            embedding_dim=cfg["embedding_dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            ffn_hidden=cfg["ffn_hidden"],
            dropout=cfg["dropout"],
        ).to(device)
        eval_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        eval_model.eval()

        df_test = load_test(variable, columns=cols)
        X_test  = torch.from_numpy(df_test[FEATURE_COLS].values.astype(np.float32))
        y_test  = df_test[label_col].values.astype(np.float32)
        del df_test

        # Inferência em batches para não estourar a memória da GPU com 47M amostras.
        # batch_size * 4: sem gradientes, ocupa ~4× menos memória que o treino.
        preds = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for s in range(0, len(X_test), cfg["batch_size"] * 4):
                chunk = X_test[s : s + cfg["batch_size"] * 4].to(device, non_blocking=True)
                preds.append(eval_model(chunk).cpu().numpy())
        y_pred = np.concatenate(preds)

        # Métricas no espaço MinMax [0,1] — consistente com 3.0, 4.0 e 5.0.
        metrics = compute_metrics(y_test.astype(np.float64), y_pred.astype(np.float64))
        save_metrics(
            metrics,
            RESULTS_6 / config_name,
            variable,
            extra_cols={
                "config":        config_name,
                "n_train":       n_tr,
                "best_val_mae":  round(best_val_mae, 6),
                "total_time_s":  round(time.time() - t0, 1),
            },
        )
        print(
            f"  [{variable}/{config_name}] TESTE —"
            f"  MAE={metrics['MAE']:.4f}"
            f"  RMSE={metrics['RMSE']:.4f}"
            f"  R²={metrics['R²']:.4f}"
            f"  r={metrics['r']:.4f}",
            flush=True,
        )

    # Sincroniza todos os ranks antes de encerrar: ranks 1..4 esperam aqui enquanto
    # rank 0 termina a avaliação no teste. Sem isso, ranks 1..4 chamariam
    # destroy_process_group antes de rank 0 terminar — o heartbeat do NCCL detecta
    # "peer saiu inesperadamente" e pode matar rank 0 no meio da avaliação.
    dist.barrier()

    # Encerra o grupo de comunicação e libera os recursos NCCL.
    # Obrigatório chamar antes do processo terminar para evitar leaks de socket.
    dist.destroy_process_group()


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_free_port() -> int:
    """
    Encontra uma porta TCP livre no host.
    Cada torchrun precisa de uma porta diferente para o processo de rendezvous
    (coordenação inicial entre os ranks). Sem isso, dois torchrun consecutivos
    poderiam tentar usar a mesma porta e o segundo falharia com 'address in use'.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _aggregate_summary() -> None:
    """Consolida todos os metrics.csv de cada config/variável em comparison.csv."""
    rows = []
    for config_name in CONFIGS:
        for var in VARIABLES:
            p = RESULTS_6 / config_name / var / "metrics.csv"
            if p.exists():
                df = pd.read_csv(p)
                df.insert(0, "variable", var)
                rows.append(df)
    if not rows:
        return

    summary = pd.concat(rows, ignore_index=True)
    out = RESULTS_6 / "comparison.csv"
    summary.to_csv(out, index=False)
    print(f"\nResumo → {out}")

    if summary["config"].nunique() > 1:
        pivot = summary.pivot_table(index="variable", columns="config", values=["MAE", "R²"])
        print("\n=== MAE por config ===")
        print(pivot["MAE"].round(4).to_string())
        print("\n=== R² por config ===")
        print(pivot["R²"].round(4).to_string())
    else:
        print(summary[["variable", "config", "MAE", "RMSE", "R²", "r"]].to_string(index=False))


# ── orquestrador ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orquestrador: para cada (variável, config), lança torchrun com todas as GPUs.

    Estratégia: sequencial por variável, DDP por config.
    - temperature/base  → torchrun (5 GPUs) → aguarda → temperature/wide → ...
    - Cada torchrun é um processo separado que gerencia 5 subprocessos internamente.
    - Usar torchrun (em vez de multiprocessing.spawn) evita os problemas de contexto
      CUDA observados no 4.0 quando múltiplos spawns coexistiam.
    """
    try:
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    if n_gpus == 0:
        print("Nenhuma GPU detectada — SNT requer GPU.", flush=True)
        sys.exit(1)

    print(f"=== 6.0 SNT (Spatial Neighbor Transformer) ===")
    print(f"GPUs: {n_gpus}")
    for cfg_name, cfg in CONFIGS.items():
        eff = cfg["batch_size"] * n_gpus
        print(
            f"  [{cfg_name}]  D={cfg['embedding_dim']}"
            f"  layers={cfg['n_layers']}  heads={cfg['n_heads']}"
            f"  batch/GPU={cfg['batch_size']:,}  efetivo={eff:,}"
        )
    print(f"\n{len(VARIABLES)} variáveis × {len(CONFIGS)} configs"
          f" = {len(VARIABLES)*len(CONFIGS)} runs  (sequencial)\n")

    RESULTS_6.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for variable in VARIABLES:
        for config_name in CONFIGS:
            # Porta nova a cada run: garante que dois torchrun consecutivos
            # não colidam no rendezvous mesmo que o OS demore a liberar a porta.
            port = _find_free_port()
            print(f"\n{'='*60}")
            print(f"  {variable:20s} / {config_name}  →  {n_gpus} GPUs  port={port}")
            print(f"{'='*60}")

            # torch.distributed.run é o módulo Python equivalente ao comando torchrun.
            # Flags importantes:
            #   --nproc_per_node N : lança N processos neste nó (1 por GPU)
            #   --master_addr      : endereço do processo coordenador (rank 0)
            #   --master_port      : porta para o rendezvous inicial entre ranks
            #   --nnodes 1         : treinamento em um único nó (servidor local)
            # O próprio script é passado como argumento — cada processo filho roda
            # __main__ com --worker, entrando em _ddp_worker().
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(n_gpus),
                "--master_addr",    "localhost",
                "--master_port",    str(port),
                "--nnodes",         "1",
                __file__,
                "--variable", variable,
                "--config",   config_name,
                "--worker",
            ]
            result = subprocess.run(cmd)

            tag = f"{variable}/{config_name}"
            if result.returncode == 0:
                print(f"  [OK  ] {tag}", flush=True)
            else:
                print(f"  [ERRO] {tag}  exit={result.returncode}", flush=True)
                failed.append(tag)

    if failed:
        print(f"\nRuns com falha: {failed}", flush=True)

    _aggregate_summary()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variable", default=None)
    parser.add_argument("--config",   default=None)
    parser.add_argument("--worker",   action="store_true",
                        help="modo DDP worker — chamado pelo torchrun, não invocar manualmente")
    args = parser.parse_args()

    if args.worker:
        # Modo worker: executado pelos processos filhos do torchrun.
        # RANK, WORLD_SIZE e LOCAL_RANK já estão nas variáveis de ambiente.
        if not args.variable or not args.config:
            sys.exit("--variable e --config são obrigatórios no modo worker")
        _ddp_worker(args.variable, args.config)
    else:
        # Modo orquestrador: executado diretamente pelo usuário.
        main()
