# Climate-Intelligence-Engine

Engine analítica para detecção de inconsistências, completamento de dados e otimização de precisão climática em estações INMET brasileiras.

---

## Instalação

### macOS

```bash
brew install --cask miniconda
conda init zsh   # ou bash
# reinicie o terminal
conda env create -f environment.yml
conda activate climate-engine
python -m ipykernel install --user --name climate-engine --display-name "Climate Engine"
```

### Linux (servidor com GPU NVIDIA)

```bash
# instalar Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda init bash && source ~/.bashrc

conda env create -f environment.yml
conda activate climate-engine
python -m ipykernel install --user --name climate-engine --display-name "Climate Engine"
```

> O `environment.yml` instala PyTorch com suporte a CUDA 12.6 via pip wheels. Compatible com drivers NVIDIA a partir do 525 (CUDA 12.x).

---

## Execução via Notebook

Abra `main.ipynb` no VS Code ou Jupyter Lab e selecione o kernel **Climate Engine**. O notebook orquestra todo o pipeline chamando cada script em sequência.

```bash
jupyter lab   # alternativa ao VS Code
```

---

## Pipeline

Execute os scripts **na ordem abaixo** com o ambiente ativado.

---

### 1.1 — Download dos dados

Baixa `stations.parquet` e `weather_measurements.parquet` do Google Drive para `data/`.

```bash
python 1.1_download_data.py
```

---

### 1.2 — Distâncias entre estações

Gera `data/station_distances.parquet` com distâncias geodésicas (Haversine), delta de altitude e azimute par-a-par entre todas as estações.

```bash
python 1.2_compute_station_distances.py
```

| Coluna | Descrição |
|---|---|
| `from_code` / `to_code` | Códigos das estações |
| `distance_km` | Distância geodésica (Haversine) em km |
| `delta_altitude_m` | Diferença de altitude em metros |
| `azimuth_deg` | Azimute em graus (0–360) |

> Requer 1.1.

---

### 1.4 — Limpeza e separação por variável

Lê `weather_measurements.parquet`, remove NaN, substitui radiação negativa por 0 e gera um parquet por variável com schema `code / time / measurement`.

```bash
python 1.4_clean_data.py
```

| Arquivo | Variável |
|---|---|
| `temperature.parquet` | Temperatura (°C) |
| `humidity.parquet` | Umidade relativa (%) |
| `rainfall.parquet` | Chuva (mm/h) |
| `global_radiation.parquet` | Radiação solar (KJ/m²) |
| `pressure.parquet` | Pressão atmosférica (hPa) |

> Requer 1.1.

---

### 1.5 — Enriquecimento com vizinhos

Para cada variável, adiciona as medições das 20 estações mais próximas disponíveis no mesmo timestamp.

```bash
python 1.5_build_neighbors.py
```

Gera `data/{variable}_neighbors.parquet` com colunas `code, time, measurement, n01…n20, d01…d20, a01…a20, azimuth01…azimuth20`.

> Requer 1.2 e 1.4.

---

### 1.6 — Scaling de features (MinMaxScaler [0,1])

Aplica MinMaxScaler com clipping em percentis [p1, p99] para robustez a outliers. Features sin/cos (azimute, hora, dia do ano) são matematicamente limitadas a [−1, 1] e não precisam de scaling.

Usa dados a partir de **2002** (primeiro ano com ≥ 15 vizinhos disponíveis para todas as variáveis). Split temporal: 80% treino / 20% teste, ordenados por timestamp.

```bash
python 1.6_scale_features.py
```

| Destino | Arquivo | Descrição |
|---|---|---|
| `data/` | `{variable}_train_scaled.parquet` | Features escaladas — treino |
| `data/` | `{variable}_test_scaled.parquet` | Features escaladas — teste |
| `models/` | `scaler_{variable}.scaler` | MinMaxScaler serializado (joblib) |

Features (79 colunas para k=15 vizinhos):

| Grupo | Colunas | Scaling |
|---|---|---|
| Medição vizinhos | `n01..n15` | MinMax [0,1] |
| Distância (km) | `d01..d15` | MinMax [0,1] |
| Delta altitude (m) | `a01..a15` | MinMax [0,1] |
| Azimute | `b01..b15_sin`, `b01..b15_cos` | Nenhum (já em [−1,1]) |
| Temporais cíclicos | `hour_sin/cos`, `doy_sin/cos` | Nenhum (já em [−1,1]) |

> Requer 1.5.

---

### 2.0 — Baseline: vizinho mais próximo

Avalia `n01` (vizinho mais próximo) como estimador direto da medição da estação-alvo no conjunto de **teste**. É o baseline mínimo — qualquer modelo treinado deve superá-lo.

```bash
python 2.0_neighbors.py
```

Métricas: MAE, RMSE, R², Bias, r (Pearson) — todas no espaço MinMax [0,1].

Gera `results/2.0_neighbors/metrics.csv`.

> Requer 1.6.

---

### 3.0 — Ridge Regression

Treina Ridge Regression por variável. Ridge (OLS + λI) resolve a multicolinearidade de `n01..n15` (temperaturas de estações vizinhas altamente correlacionadas) estabilizando XᵀX sem sacrificar precisão.

XᵀX (80×80) é computado **uma única vez** sobre todo o treino em memória. A busca do melhor λ é negligível — cada candidato é apenas um `solve` em 80×80.

```bash
python 3.0_linear_regression.py
```

| Parâmetro | Valor |
|---|---|
| λ candidatos | 0.001, 0.01, 0.1, 1.0, 10.0, 100.0 |
| Validação | últimos 10% do treino (temporal) |
| Critério | menor MAE na validação |

| Destino | Arquivo | Descrição |
|---|---|---|
| `results/3.0_linear_regression/{variable}/` | `model.npy` | Coeficientes β |
| `results/3.0_linear_regression/{variable}/` | `metrics.csv` | MAE, RMSE, R², Bias, r, melhor λ |
| `results/3.0_linear_regression/` | `metrics.csv` | Resumo por variável |

> Requer 1.6.

---

### 4.0 — Rede Neural Densa (MLP)

MLP com arquitetura expand-then-compress otimizada para gap-filling. Detecta automaticamente GPUs disponíveis; com múltiplas GPUs, processa uma variável por GPU em paralelo.

```bash
python 4.0_dense_layer.py
```

**Arquitetura:**

```
Input(79) → Linear(256) → BN → GELU → Dropout(0.20)
          → Linear(512) → BN → GELU → Dropout(0.20)
          → Linear(256) → BN → GELU → Dropout(0.20)
          → Linear(128) → BN → GELU → Dropout(0.20)
          → Linear(64)  → BN → GELU → Dropout(0.10)
          → Linear(1)
```

| Parâmetro | Valor |
|---|---|
| Otimizador | AdamW (lr=1e-3, weight_decay=1e-4) |
| Loss | Huber (delta=0.05) |
| Scheduler | ReduceLROnPlateau (fator=0.5, patience=10) |
| Early stop | patience=25 épocas sem melhora no val MAE |
| Val split | últimos 10% do treino (temporal) |
| Max épocas | 750 |
| Precisão | float32 + AMP (mixed precision) na GPU |

3 configurações treinadas em paralelo por variável (base / wide / xl).

| Destino | Arquivo | Descrição |
|---|---|---|
| `models/` | `{variable}_{config}_dense.pt` | Melhor state_dict (menor val MAE) |
| `results/4.0_dense_layer/{variable}/` | `training_log_{config}.csv` | loss/MAE por época |
| `results/4.0_dense_layer/{variable}/` | `metrics.csv` | Métricas no teste |
| `results/4.0_dense_layer/` | `metrics.csv` | Resumo por variável |

> Requer 1.6. GPU obrigatória.

---

### 5.0 — Random Forest (LightGBM RF mode, GPU)

Random Forest via LightGBM no modo RF com histogramas acelerados por GPU. Uma variável por GPU em paralelo via `CUDA_VISIBLE_DEVICES` + `subprocess`.

```bash
python 5.0_random_forest.py
```

| Parâmetro | Valor |
|---|---|
| Árvores | 2000 (com early stopping em 100) |
| num_leaves | 511 |
| feature_fraction_bynode | 0.33 (~26 de 79 features por nó) |
| bagging_fraction | 0.80 |
| min_child_samples | 10 |
| Objetivo | regression (MSE); rainfall usa tweedie (power=1.5) |
| Val split | últimos 10% do treino (temporal) |

| Destino | Arquivo | Descrição |
|---|---|---|
| `models/` | `{variable}_rf.lgb` | Modelo LightGBM serializado |
| `results/5.0_random_forest/{variable}/` | `metrics.csv` | Métricas no teste |
| `results/5.0_random_forest/{variable}/` | `feature_importance.csv` | Importância por feature (gain + split) |
| `results/5.0_random_forest/` | `metrics.csv` | Resumo por variável |

> Requer 1.6. GPU recomendada (fallback automático para CPU).

---

## Estrutura de diretórios

```
Climate-Intelligence-Engine/
├── data/                        # parquets gerados pelo pipeline
├── models/                      # scalers e pesos dos modelos
├── results/                     # métricas por script
│   ├── 2.0_neighbors/
│   ├── 3.0_linear_regression/
│   ├── 4.0_dense_layer/
│   └── 5.0_random_forest/
├── main.ipynb                   # notebook principal
├── utils.py                     # funções compartilhadas
├── environment.yml
└── 1.1_download_data.py … 5.0_random_forest.py
```

---

## Recriar o ambiente

```bash
conda env remove -n climate-engine
conda env create -f environment.yml
```
