"""
This is a boilerplate pipeline 'linear_model'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_test_split_df, get_results


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_test_split_df,
                inputs=["model_data", "parameters"],
                outputs=dict(
                    X_train_linear="X_train_linear", X_test_linear="X_test_linear"
                ),
                name="train_test_split__linear",
            ),
            node(
                func=get_results,
                inputs=["X_train_linear", "X_test_linear", "parameters"],
                outputs=dict(results="task1_results", predictions="task1_predictions"),
                name="train_linear_models_task1",
            ),
        ]
    )
