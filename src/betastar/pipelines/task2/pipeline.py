"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_df,
    features_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_df,
                inputs=["model_data", "parameters"],
                outputs=dict(
                    X_train="X_train",
                    X_test="X_test",
                    y_train="y_train",
                    y_test="y_test",
                ),
                name="train_test_split_df",
            ),
            node(
                func=features_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "parameters",
                    "params:cutoffs",
                ],
                outputs="task2_results",
                name="train__classifiers",
            ),
        ]
    )
