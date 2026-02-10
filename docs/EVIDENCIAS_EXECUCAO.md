# Evidências de Execução — Projeto MLOps (Aulas 1–4)

**Aluna:** Raynane Camillo Carvalho  
**Entrega:** 16/02/2026  

> Objetivo: documentar evidências da execução do pipeline (treino, tracking no MLflow, testes) e do app Streamlit (local e/ou Docker), incluindo os comandos de versionamento no Git.

---

## 0) Ambiente e dependências

### 0.1) Criar venv
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 0.2) Instalar dependências
```bash
pip install -r requirements.txt
```

**Evidência (print):**
- [ ] Print do terminal com o `pip install ...` finalizado

---

## 1) Aula 1 — EDA em notebook

Arquivo: `notebooks/EDA_diamond.ipynb`

**Evidência (prints):**
- [ ] print do notebook com: distribuição de `price`, correlação, insights
- [ ] célula mostrando o dataset `diamonds` carregado

---

## 2) Aula 2 — Estruturação do projeto + módulo de dados

Arquivo: `src/data.py`

### 2.1) Rodar teste rápido do split
```bash
python teste_aula_02.py
```

**Evidência (colar output):**
```text
# cole aqui o output do terminal
```

---

## 3) Aula 3 — Pipeline de treino + MLflow + testes

### 3.1) Treinar o modelo (MLflow local em arquivo)
```bash
python train.py
```

**Evidências:**
- [ ] print do terminal mostrando: Run ID, Tracking URI, métricas (MAE/RMSE/R2)
- [ ] confirmação do arquivo gerado: `models/diamond_price_model.joblib`

### 3.2) Abrir a UI do MLflow (modo simples)
```bash
mlflow ui --backend-store-uri ./mlruns
```

**Evidências:**
- [ ] print do browser com a UI do MLflow
- [ ] experimento `diamond_price_experiment` visível

### 3.3) (Opcional) MLflow server em http://localhost:5000
```bash
mlflow server --host 0.0.0.0 --port 5000
```

Em outro terminal:
```bash
# Windows
set MLFLOW_TRACKING_URI=http://localhost:5000
python train.py

# Linux/Mac
export MLFLOW_TRACKING_URI=http://localhost:5000
python train.py
```

**Evidências:**
- [ ] print da UI em `http://localhost:5000`

### 3.4) Rodar testes
```bash
pytest
```

**Evidência (colar output):**
```text
# cole aqui o output do pytest
```

---

## 4) Aula 4 — App Streamlit + Docker

### 4.1) Rodar o app localmente
```bash
streamlit run app/streamlit_app.py
```

**Evidências:**
- [ ] print do terminal com a URL
- [ ] print da tela do app (com a personalização: nome + logo + layout)

### 4.2) Rodar com Docker
```bash
docker compose up --build
```

Acessar: http://localhost:8501

**Evidências:**
- [ ] print do terminal com container subindo
- [ ] print do app no browser

---

## 5) Versionamento no Git (comandos e histórico)

### 5.1) Inicializar repositório e primeiro commit
```bash
git init
git add .
git commit -m "chore: estrutura inicial do projeto (aulas 1-2)"
```

### 5.2) Commits sugeridos (exemplo)
```bash
git commit -am "feat: pipeline de treino + MLflow + testes (aula 3)"
# depois
 git commit -am "feat: app Streamlit + Docker (aula 4)"
# depois
 git commit -am "style: personalizacao do app (assinatura)"
```

### 5.3) Publicar no GitHub
```bash
git remote add origin <URL_DO_SEU_REPO>
git branch -M main
git push -u origin main
```

**Evidências:**
- [ ] print do `git log --oneline --decorate -5`
- [ ] print do GitHub mostrando os commits

---

## 6) Link de entrega

- Link do repositório: <COLE AQUI>

