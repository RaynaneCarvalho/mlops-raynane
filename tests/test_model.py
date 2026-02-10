from sklearn.pipeline import Pipeline

from src.data import load_diamonds, split_features_target
from src.model import build_model


def test_build_model_returns_pipeline():
    df = load_diamonds()
    X, _ = split_features_target(df)

    pipeline = build_model(df_sample=X)

    assert isinstance(pipeline, Pipeline)
    assert "preprocess" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_can_fit():
    df = load_diamonds().sample(500, random_state=42)
    X, y = split_features_target(df)

    pipeline = build_model(df_sample=X, max_depth=3)

    pipeline.fit(X, y)

    assert pipeline is not None
