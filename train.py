"""Treino do modelo de preço de diamantes.

Este script foi pensado para funcionar tanto:

1) Com **MLflow local em arquivo** (default) — não precisa do servidor rodando.
2) Com **MLflow server** (http://localhost:5000) — basta exportar a variável
   `MLFLOW_TRACKING_URI` ou passar `--tracking-uri`.

Isso deixa o projeto mais reprodutível (e permite rodar testes/CI sem depender de
serviços externos).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.data import train_test_split_diamonds
from src.evaluate import regression_metrics
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina e versiona um modelo (diamonds)")

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        help=(
            "Tracking URI do MLflow. Ex.: http://localhost:5000 ou file:./mlruns. "
            "Default: file:./mlruns"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "diamond_price_experiment"),
        help="Nome do experimento MLflow",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_OUTPUT_PATH", "models/diamond_price_model.joblib"),
        help="Caminho para salvar a cópia local do modelo (para o Streamlit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # garante pasta local de modelos
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # dados
    X_train, X_test, y_train, y_test = train_test_split_diamonds(test_size=args.test_size)

    # pipeline
    pipeline = build_model(df_sample=X_train, max_depth=args.max_depth)

    with mlflow.start_run() as run:
        # parâmetros
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("tracking_uri", args.tracking_uri)

        # treino
        pipeline.fit(X_train, y_train)

        # predição
        y_pred = pipeline.predict(X_test)

        # métricas
        metrics = regression_metrics(y_test, y_pred)
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        # assinatura
        signature = infer_signature(X_test, y_pred)

        # log do modelo no MLflow
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="diamond_price_model",
            signature=signature,
        )

        # cópia local do modelo para consumo no app
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="exported_model")

        print("Treino concluído.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking URI: {args.tracking_uri}")
        print(f"Modelo salvo em: {model_path}")
        print(f"MAE:  {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R2:   {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
