from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_graph,
    node2vec_embedding,
    concat_data,
    preprocessing,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_graph,
                inputs=["graph"],
                outputs=["graph_pre","graph_struct"],
                name="python_graph_preprocessing",
            ),
            node(
                func=node2vec_embedding,
                inputs=["graph_pre", "parameters"],
                outputs="embedded_graph_node2vec",
                name="node2vec_embedding_preprocessing",
            ),
            node(
                func=concat_data,
                inputs=["data", "parameters", 
                        "embedded_graph_node2vec",
                        "embedded_graph_struc2vec"],
                outputs="concat_data",
                name="concat_data",
            ),
            node(
                func=preprocessing,
                inputs="concat_data",
                outputs="model_data",
                name="preprocessing",
            ),
        ]
    )
