"""Aplicação Streamlit — Previsão de preço de diamantes.

Personalização do trabalho (assinatura):
- Layout + identidade visual com o nome da autora
- Logo em SVG embutido (sem depender de Pillow)
- Fallback inteligente para carregar modelo local (joblib) ou via MLflow
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_AUTHOR = "Raynane Camillo Carvalho"
APP_TAGLINE = "MLOps — Impacta | Projeto individual"
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/diamond_price_model.joblib")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: radial-gradient(circle at 10% 10%, #1f2b4d 0%, #0b1020 45%, #070a12 100%);
          }
          /* deixa o conteúdo mais "card" */
          [data-testid="stVerticalBlock"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 18px 18px 6px 18px;
          }
          /* remove o menu e o footer */
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _logo_svg() -> str:
    # SVG simples: diamante + iniciais
    return """
    <svg width="120" height="36" viewBox="0 0 120 36" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="logo">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#7dd3fc"/>
          <stop offset="100%" stop-color="#a78bfa"/>
        </linearGradient>
      </defs>
      <g fill="none" fill-rule="evenodd">
        <path d="M18 3 L29 12 L18 33 L7 12 Z" fill="url(#g)" opacity="0.95"/>
        <path d="M7 12 L29 12" stroke="rgba(255,255,255,0.5)"/>
        <path d="M18 3 L18 33" stroke="rgba(255,255,255,0.35)"/>
        <text x="40" y="24" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI" font-size="14" fill="rgba(255,255,255,0.92)">
          RC • Diamonds
        </text>
      </g>
    </svg>
    """.strip()


@st.cache_resource
def load_model_local(model_path: str):
    """Carrega o modelo salvo localmente (joblib)."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Rode: `python train.py` para gerar o arquivo."
        )
    return joblib.load(path)


@st.cache_resource
def load_model_mlflow(model_uri: str):
    """Carrega modelo via MLflow (opcional)."""
    import mlflow.pyfunc  # import local para não quebrar caso mlflow não esteja instalado

    return mlflow.pyfunc.load_model(model_uri)


def get_model():
    """Escolhe a melhor fonte do modelo.

    Prioridade:
    1) MLflow (se USE_MLFLOW_MODEL=true)
    2) arquivo local joblib
    """
    use_mlflow = os.getenv("USE_MLFLOW_MODEL", "false").lower() in {"1", "true", "yes"}

    if use_mlflow:
        # exemplo: models:/diamonds_price_model@champion
        model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/diamonds_price_model@champion")
        try:
            return load_model_mlflow(model_uri), f"MLflow ({model_uri})"
        except Exception as e:  # noqa: BLE001
            st.warning(
                f"Não foi possível carregar via MLflow. Caindo para modelo local. Detalhe: {e}"
            )

    return load_model_local(DEFAULT_MODEL_PATH), f"Local ({DEFAULT_MODEL_PATH})"


def build_input_form() -> pd.DataFrame:
    st.subheader("Informe as características do diamante")

    col1, col2 = st.columns(2)
    with col1:
        carat = st.number_input("carat", min_value=0.0, max_value=5.0, value=0.7, step=0.01)
        depth = st.number_input("depth", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
        table = st.number_input("table", min_value=40.0, max_value=80.0, value=57.0, step=0.1)

    with col2:
        x = st.number_input("x (comprimento)", min_value=0.0, max_value=15.0, value=5.5, step=0.1)
        y = st.number_input("y (largura)", min_value=0.0, max_value=15.0, value=5.5, step=0.1)
        z = st.number_input("z (altura)", min_value=0.0, max_value=15.0, value=3.4, step=0.1)

    cut = st.selectbox("cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=3)
    color = st.selectbox("color", ["D", "E", "F", "G", "H", "I", "J"], index=3)
    clarity = st.selectbox(
        "clarity",
        ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
        index=3,
    )

    data = pd.DataFrame(
        {
            "carat": [float(carat)],
            "depth": [float(depth)],
            "table": [float(table)],
            "x": [float(x)],
            "y": [float(y)],
            "z": [float(z)],
            "cut": [str(cut)],
            "color": [str(color)],
            "clarity": [str(clarity)],
        }
    )

    # ordem esperada
    expected_cols = ["carat", "depth", "table", "x", "y", "z", "cut", "color", "clarity"]
    return data[expected_cols]


def main() -> None:
    st.set_page_config(page_title="Diamonds • MLOps", page_icon="💎")
    _inject_css()

    # Sidebar (assinatura)
    st.sidebar.markdown(_logo_svg(), unsafe_allow_html=True)
    st.sidebar.markdown(f"**{APP_AUTHOR}**")
    st.sidebar.caption(APP_TAGLINE)
    st.sidebar.divider()
    st.sidebar.write(
        "Este app consome um modelo treinado com o dataset `diamonds` (seaborn) e prevê o preço."
    )
    st.sidebar.info(
        "Dica: rode `python train.py` para gerar/atualizar o modelo em `models/diamond_price_model.joblib`."
    )

    st.title("💎 Previsão de preço de diamantes")
    st.caption("App de inferência (deploy) — Aula 4")

    with st.expander("Sobre este projeto", expanded=False):
        st.write(
            "- Aula 1: EDA em notebook\n"
            "- Aula 2: estruturação do projeto + módulo de dados\n"
            "- Aula 3: pipeline de treino + MLflow + testes\n"
            "- Aula 4: app Streamlit em Docker (inferência)\n"
        )

    model, model_source = get_model()
    st.caption(f"Modelo carregado de: **{model_source}**")

    data = build_input_form()

    if st.button("Prever preço", type="primary"):
        try:
            prediction = float(model.predict(data)[0])
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro ao executar predição: {e}")
            st.stop()

        st.subheader("Resultado")
        st.metric(label="Preço estimado (USD)", value=f"${prediction:,.2f}")
        st.caption("Obs.: valor estimado apenas para fins didáticos.")


if __name__ == "__main__":
    main()
