import pandas as pd

from src.data import (
    load_diamonds,
    split_features_target,
    train_test_split_diamonds,
)


def test_load_diamonds_returns_dataframe():
    df = load_diamonds()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "price" in df.columns


def test_split_features_target():
    df = load_diamonds()
    X, y = split_features_target(df)

    assert "price" not in X.columns
    assert len(X) == len(y)
    assert y.name == "price"


def test_train_test_split_diamonds_shapes():
    X_train, X_test, y_train, y_test = train_test_split_diamonds()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
