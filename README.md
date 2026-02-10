# Projeto MLOps – Previsão de Preço de Diamantes

**Autora:** Raynane Camillo Carvalho  \
**Entrega:** 16/02/2026 (trabalho individual)  \

## Personalização (assinatura)

Além de executar as atividades práticas das aulas 1 a 4, o app Streamlit foi personalizado com uma identidade visual própria:

- nome da autora no sidebar
- logo em SVG embutido (sem dependências extras)
- layout escuro com fundo em gradiente
- carregamento de modelo com fallback (MLflow opcional → modelo local)

Este repositório faz parte da disciplina **MLOps – Running ML in Production Environments**.

O objetivo do projeto é mostrar, passo a passo, como evoluir de um fluxo manual em notebooks para um projeto de Machine Learning organizado, reprodutível e pronto para automação, seguindo boas práticas de MLOps.

---

## Contexto até aqui

Na **Aula 1**, trabalhamos com um fluxo típico de ciência de dados:

- EDA em notebook  
- Treino e inferência manual  
- Uso inicial do MLflow para registrar experimentos  

Esse fluxo funciona, mas não escala e não é fácil de repetir.

Na **Aula 2**, o foco foi **organizar o projeto e estruturar o ciclo de dados**, preparando o terreno para automação e evolução do pipeline nas próximas aulas.

---

## O que foi feito na Aula 2

Nesta etapa, o projeto passou por uma reorganização importante:

- Criação de um repositório GitHub  
- Estruturação do projeto em pastas claras  
- Separação da lógica de dados em um módulo Python  
- Criação de um ponto único para carregar e dividir os dados  
- Integração do notebook com o código do projeto  

O notebook deixou de ser responsável por toda a lógica de dados e passou a **consumir funções reutilizáveis**.

---

## O que foi feito na Aula 3

Na Aula 3, o projeto evoluiu do preparo de dados para um **pipeline completo de modelagem**.

Foram implementados:

- separação da lógica de modelagem em módulos Python  
- pipeline de pré-processamento e treino com scikit-learn  
- script de treino executável via linha de comando  
- avaliação padronizada de métricas de regressão  
- experiment tracking completo com MLflow  
- versionamento de modelos no MLflow Model Registry  
- testes automatizados com pytest  

A partir deste ponto, o modelo deixa de depender do notebook e passa a ser tratado
como um **artefato versionado e rastreável**.


## Estrutura atual do projeto

```text
impacta_mlops/
│
├── notebooks/
│   └── eda_diamonds.ipynb
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── __init__.py
│   └── evaluate.py
│
├── app/
│
├── tests/
│   ├── test_data.py
│   ├── __init__.py
│   ├── test_model.py
│   └── test_train.py
│
├── models/
│   └── diamond_price_model.joblib
│
├── requirements.txt
├── train.py
├── pytest.ini
├── main.py
├── README.md
└── .gitignore

```

---

## Módulo de dados

O arquivo `src/data.py` centraliza a lógica relacionada aos dados:

- carregamento do dataset `diamonds` do seaborn  
- separação de features e target  
- divisão em treino e teste  

Isso garante que todos usem **o mesmo processo de preparação**, evitando inconsistências entre notebooks e scripts.

---

## Uso no notebook

O notebook agora utiliza diretamente o módulo de dados:

```python
from src.data import train_test_split_diamonds

X_train, X_test, y_train, y_test = train_test_split_diamonds()
```

Com isso:

- a lógica de dados fica centralizada  
- o notebook fica mais simples  
- o código se torna reutilizável  
- o projeto começa a ganhar reprodutibilidade  

---

## Ambiente de desenvolvimento

Recomenda-se o uso de ambiente virtual.

Criar e ativar o ambiente:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

---

### – Pipeline de treino


O treino do modelo é executado via script Python, sem dependência de notebook.

Execução padrão:

```bash
python train.py
```

Por padrão, o tracking do MLflow é salvo localmente em `./mlruns` (não precisa de servidor).

Para abrir a UI nesse modo:

```bash
mlflow ui --backend-store-uri ./mlruns
```


É possível ajustar hiperparâmetros via linha de comando:

```bash
python train.py --max_depth 3
```

Durante o treino, são registrados no MLflow:

- parâmetros
- métricas
- artefatos
- modelo treinado

Isso permite comparar experimentos e versionar modelos de forma consistente.

---

### Testes automatizados
O projeto possui testes básicos para garantir a estabilidade do pipeline.

Execução dos testes:

```bash
pytest
```

Os testes cobrem:
carregamento e split dos dados
construção do pipeline de modelagem
execução completa do script de treino
Esses testes ajudam a garantir que refatorações não quebrem o fluxo principal.

# Aula 4 – Deploy, Operação e Ciclo Completo de MLOps

Esta aula finaliza o projeto e fecha o ciclo completo de **MLOps**, mostrando como um modelo treinado e versionado pode ser **consumido por uma aplicação real**, com controle de ambiente e foco em operação.

---

## Contexto da Aula 4

Até o final da Aula 3, o projeto já possuía:

- dados organizados e padronizados  
- pipeline de treino estruturado  
- versionamento de código e experimentos  
- rastreabilidade completa com MLflow  
- modelo registrado e validado  

Na **Aula 4**, o foco deixa de ser o treino do modelo e passa a ser **o uso do modelo**, simulando um cenário real de deploy.

---

## Arquitetura adotada

Para manter o ambiente simples, estável e didático, foi adotada a seguinte arquitetura:

- **MLflow rodando localmente (host)**  
- responsável por tracking, registry e histórico  

- **Aplicação Streamlit rodando em Docker**  
- consome o modelo registrado no MLflow (cópia local) 
- executa inferência em tempo real  

---

## Aplicação de predição

Foi criada uma aplicação em **Streamlit** para previsão do preço de diamantes.

A aplicação:

- carrega o modelo campeão localmente, mas no futuro queremos direto do **MLflow Model Registry**  
- recebe dados do usuário via formulário  
- executa inferência em tempo real  
- retorna a previsão de preço  

O modelo **não é treinado no app**, apenas consumido, reforçando a separação entre treino e inferência.

---

## Dockerização do app

A aplicação Streamlit foi empacotada em um container Docker, garantindo:

- isolamento de ambiente  
- reprodutibilidade  
- facilidade de execução em qualquer máquina  

O Docker é utilizado apenas onde agrega valor, sem forçar a containerização de todos os componentes do pipeline.

---

## Execução do projeto

### 1. MLflow (2 opções)

**Opção A — Sem servidor (mais simples):**

1) treine o modelo:

```bash
python train.py
```

2) abra a UI apontando para `./mlruns`:

```bash
mlflow ui --backend-store-uri ./mlruns
```

**Opção B — Com servidor local (http://localhost:5000):**

1) suba o servidor MLflow:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

2) em outro terminal, exporte a variável e treine:

```bash
set MLFLOW_TRACKING_URI=http://localhost:5000
python train.py
```

Nesse modo, a UI ficará em:
http://localhost:5000
---

### 2. Subir a aplicação com Docker

Em outro terminal, execute:
`docker compose up --build`
A aplicação Streamlit ficará disponível em:
http://localhost:8501
---
## Ciclo completo de MLOps
Ao final da Aula 4, o projeto percorre todas as etapas do ciclo de MLOps:
- exploração e entendimento dos dados
- organização do projeto
- treino estruturado e versionado
- rastreabilidade de experimentos
- registro de modelos
- consumo do modelo em aplicação
- controle de ambiente e execução

O foco do curso não é apenas treinar modelos, mas entender como mantê-los vivos, confiáveis e utilizáveis.

---

## Encerramento
Este projeto representa uma visão prática e realista de MLOps, mostrando que os principais desafios não estão apenas no modelo, mas em:
- ambiente
- versionamento
- rastreabilidade
- deploy
- operação

---

# ToDo
- mlflow no docker
- rever interface gráfica
- automatizar testes no github

