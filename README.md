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
