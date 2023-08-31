"""
This is a boilerplate pipeline 'embeddings'
generated using Kedro 0.18.11
"""

import logging
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

from typing import Any, Dict
from node2vec import Node2Vec as n2v
from sklearn.impute import SimpleImputer
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


logger = logging.getLogger(__name__)

PROJECT_PATH = Path.cwd()


# PREPROCESSING GRAPH
def preprocess_graph(data, parameters: Dict[str, Any]) -> pd.DataFrame:
    """check if the graph is zero based
    use parameters.yml file to check if the graph is start from zero
    zero_based_graph: false -> graph start from 1 (Julia case)
    zero_based_graph: true -> graph start from 0 (Python case)

    Automatise it for the future!!!
    check if min value is 0 or 1

    """
    if not parameters["zero_based_graph"]:
        data = _change_first_element(data)
    return data


def _change_first_element(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Your graph edges start from 1, let's change that")
    data["in"] = data["in"] - 1
    data["out"] = data["out"] - 1
    return data


def node2vec_embedding(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    data: pd.DataFrame
        The data to be embedded with node2vec algorithm.
        It should be a pandas dataframe with two columns: in and out
        First node should be 0.

    parameters: Dict[str, Any]

    parameter for node2vec algorithm.
    seed = 123
    workers = 1 - must be 1 for reproducibility

    """
    logger.info(
        f"Using node2vec to embed the graph \
        with {parameters['node2vec']['dimensions']} dimensions"
    )
    if "node2vec" not in parameters["embeddings"]:
        return pd.DataFrame()

    G = _load_networkx_graph(data)
    g_emb = n2v(G, **parameters["node2vec"])
    mdl = g_emb.fit()
    emb_df = pd.DataFrame([mdl.wv.get_vector(str(n)) for n in G.nodes()], index=G.nodes)
    emb_df["index"] = emb_df.index
    emb_df.sort_values(by="index", inplace=True)
    emb_df.drop(columns="index", inplace=True)
    names = {
        x: "emb_node2vec_" + str(x) for x in range(parameters["node2vec"]["dimensions"])
    }
    emb_df.rename(columns=names, inplace=True)
    return emb_df


def _load_networkx_graph(X: pd.DataFrame):
    return nx.from_pandas_edgelist(df=X, source="in", target="out")


def concat_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Concatenation for our features, embeddings and structural embeddings"""
    emb_list = parameters["embeddings"]
    emb_files = [f"embedded_graph_{i}" for i in emb_list]
    metadata = bootstrap_project(PROJECT_PATH)
    with KedroSession.create(
        package_name=metadata.package_name,
        project_path=PROJECT_PATH,
        env=parameters["env"],
    ) as session:
        context = session.load_context()
        catalog = context.catalog

    for ix, emb in enumerate(emb_list):
        if ix == 0:
            if emb == "struc2vec":
                df_emb = catalog.load(emb_files[ix])
                df_emb = _preprocess_struc(df_emb)
            else:
                df_emb = catalog.load(emb_files[ix])
            emb_con = pd.concat([data, df_emb], axis="columns")
        else:
            if emb == "struc2vec":
                df_emb = catalog.load(emb_files[ix])
                df_emb = _preprocess_struc(df_emb)
            else:
                df_emb = catalog.load(emb_files[ix])
            emb_con = pd.concat([emb_con, df_emb], axis="columns")

    return emb_con


def _preprocess_struc(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing for our structural embeddings"""
    X = df.copy()
    nr = X.columns
    new_names = {
        k: v
        for k, v in zip(
            nr, ["n_id"] + [f"emb_struc2vec_{i}" for i in range(1, len(nr))]
        )
    }
    X.rename(columns=new_names, inplace=True)
    nr = [float(i) for i in nr]
    nr[0] = int(nr[0])
    X.loc[len(X)] = nr
    X.sort_values(by="n_id", inplace=True)
    X.drop(columns=["n_id"], inplace=True)

    return X.reset_index(drop=True)


def preprocessing(data):
    """Preprocessing for our data, we drop the missing
    values and fill the rest with 0
    """
    df = data.copy()
    df.dropna(subset=["target"], inplace=True)
    si = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)

    return pd.DataFrame(si.fit_transform(df), columns=data.columns)
