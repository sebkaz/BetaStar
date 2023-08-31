"""
This is a boilerplate pipeline 'feature_importance'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_random_forest_classifier,
    pi_forest,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_random_forest_classifier,
                inputs=["X_train", "y_train", "params:random_forest"],
                outputs="forest_FI",
                name="train_random_forest_classifier_full",
            ),
            node(
                func=pi_forest,
                inputs=["X_test", "y_test", "forest_FI"],
                outputs="models_results_pi",
                name="permutation_importance_sklearn",
            ),
        ]
    )
