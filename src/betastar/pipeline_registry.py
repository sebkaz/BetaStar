"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

# from betastar.pipelines import linear_model
from betastar.pipelines import embeddings
from betastar.pipelines import task1
from betastar.pipelines import task2
from betastar.pipelines import task3


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    embeddings_pipeline = embeddings.create_pipeline()
    task1_pipeline = task1.create_pipeline()
    task2_pipeline = task2.create_pipeline()
    task3_pipeline = task3.create_pipeline()

    return {
        "emb": embeddings_pipeline,
        "t1": task1_pipeline,
        "t2": task2_pipeline,
        "t3": task3_pipeline,
        "__default__": task1_pipeline,
    }
