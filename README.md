# Climate-Intelligence-Engine
Engine analítica para detecção de inconsistências, completamento de dados e otimização de precisão climática.

---

## Instalação e Execução

### Pré-requisitos

- macOS com [Homebrew](https://brew.sh) instalado
- Git

---

### Passo 1 — Instalar o Miniconda

```bash
brew install --cask miniconda
```

Após a instalação, inicialize o conda no seu shell:

```bash
conda init zsh   # ou conda init bash, dependendo do seu shell
```

Reinicie o terminal para as mudanças entrarem em vigor.

---

### Passo 2 — Clonar o repositório

```bash
git clone <url-do-repositorio>
cd Climate-Intelligence-Engine
```

---

### Passo 3 — Criar o ambiente conda

```bash
conda env create -f environment.yml
```

---

### Passo 4 — Ativar o ambiente

```bash
conda activate climate-engine
```

---

### Passo 5 — Registrar o kernel no Jupyter

```bash
python -m ipykernel install --user --name climate-engine --display-name "Climate Engine"
```

---

### Passo 6 — Executar o notebook

**Via VS Code** — abra `main.ipynb` e selecione o kernel **Climate Engine** no canto superior direito.

**Via Jupyter Lab** — execute no terminal:

```bash
jupyter lab
```

Depois abra `main.ipynb` no navegador.

---

## Pipeline de Dados

Com o ambiente ativado (`conda activate climate-engine`), execute os scripts na ordem abaixo.

---

### 1.1 — Download dos dados

Baixa os arquivos `stations.parquet` e `weather_measurements.parquet` do Google Drive para a pasta `data/`.

```bash
python 1.1_download_data.py
```

---

### 1.2 — Cálculo de distâncias entre estações

Gera o arquivo `data/station_distances.parquet` com as distâncias par-a-par entre todas as estações meteorológicas.

```bash
python 1.2_compute_station_distances.py
```

Colunas geradas:

| Coluna | Descrição |
|---|---|
| `from_code` | Código da estação de origem |
| `to_code` | Código da estação de destino |
| `distance_km` | Distância geodésica (Haversine) em km |
| `delta_altitude_m` | Diferença de altitude em metros |
| `effective_distance_km` | Distância 3D ponderada (Haversine + altitude) |

> Requer que o Passo 1.1 tenha sido executado antes.

---

### 1.4 — Limpeza e separação das variáveis

Lê `weather_measurements.parquet` e gera um parquet por variável com a estrutura `code / time / measurement`, removendo todos os NaN.
Radiação solar negativa é substituída por 0 (ausência de luz).

```bash
python 1.4_clean_data.py
```

Arquivos gerados em `data/`:

| Arquivo | Variável |
|---|---|
| `temperature.parquet` | Temperatura (°C) |
| `humidity.parquet` | Umidade relativa (%) |
| `rainfall.parquet` | Chuva (mm/h) |
| `global_radiation.parquet` | Radiação solar (KJ/m²) |
| `pressure.parquet` | Pressão atmosférica (hPa) |

> Requer que o Passo 1.1 tenha sido executado antes.

---

### Atualizar dependências

Se você instalar novos pacotes e quiser salvar no `environment.yml`:

```bash
conda env export --no-builds > environment.yml
```

Para recriar o ambiente do zero:

```bash
conda env remove -n climate-engine
conda env create -f environment.yml
```
