from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline


def build_preprocessor(df_sample: pd.DataFrame) -> ColumnTransformer:
    """Cria o pré-processador com OneHotEncoder para colunas categóricas."""
    cat_cols = df_sample.select_dtypes(include=["object", "category"]).columns
    num_cols = df_sample.select_dtypes(exclude=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return preprocessor


def build_model(
    df_sample: pd.DataFrame,
    max_depth: Optional[int] = 5,
) -> Pipeline:
    """Cria o pipeline completo: pré-processador + modelo."""
    preprocessor = build_preprocessor(df_sample)

    regressor = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )
    return pipeline
