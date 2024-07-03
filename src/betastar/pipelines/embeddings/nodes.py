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
from fastnode2vec import Node2Vec as n2v
from fastnode2vec import Graph
from sklearn.impute import SimpleImputer
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


logger = logging.getLogger(__name__)

PROJECT_PATH = Path.cwd()


# PREPROCESSING GRAPH
def preprocess_graph(data: pd.DataFrame) -> pd.DataFrame:
    """check if the graph is zero based
    if not, change it to zero based
    I assume that it starts from 1 (like in julia)
    
    Params: 
    ------

    data: pd.DataFrame with edges two columns ('in', 'out')

    return
    ------
    modified (or not) pd.DataFrame 
    """
    if not ((0 in set(data["in"])) or (0 in set(data["out"]))):
        logger.info("Your graph edges start from 1, let's change that")
        data = data.apply(lambda x: x-1)
    return data, data


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
        f"Using fast node2vec to embed the graph \
        with {parameters['node2vec']['dim']} dimensions"
    )
    if "node2vec" not in parameters["embeddings"]:
        return pd.DataFrame()
    edges = list(zip(data['in'], data['out']))
    graph = Graph(edges, directed=False, weighted=False)
    nv = n2v(graph, **parameters["node2vec"])
    nv.train(epochs=100, verbose=True)
    emb_df = pd.DataFrame([nv.wv[i] for i in range(len(nv.wv))])
    names = {
        x: "emb_node2vec_" + str(x) for x in range(parameters["node2vec"]["dim"])
    }
    emb_df.rename(columns=names, inplace=True)
    return emb_df


def concat_data(data: pd.DataFrame,
                parameters: Dict[str, Any],
                n2vdata : pd.DataFrame,
                s2vdata : pd.DataFrame,
                ) -> pd.DataFrame:
    """Concatenation for our features, embeddings and structural embeddings"""
    # emb_list = parameters["embeddings"]

    if ('node2vec' in parameters['embeddings']) and (not n2vdata.empty):
        data = pd.concat([data, n2vdata], axis="columns")

    if ('struc2vec' in parameters['embeddings']) and (not s2vdata.empty):
        s2vdata = _preprocess_struc(s2vdata)
        data = pd.concat([data, s2vdata], axis="columns")

    return data
    # emb_files = [f"embedded_graph_{i}" for i in emb_list]
    # metadata = bootstrap_project(PROJECT_PATH)
    # with KedroSession.create(
    #     package_name=metadata.package_name,
    #     project_path=PROJECT_PATH,
    #     env=parameters["env"],
    # ) as session:
    #     context = session.load_context()
    #     catalog = context.catalog

    # for ix, emb in enumerate(emb_list):
    #     if ix == 0:
    #         if emb == "struc2vec":
    #             df_emb = catalog.load(emb_files[ix])
    #             df_emb = _preprocess_struc(df_emb)
    #         else:
    #             df_emb = catalog.load(emb_files[ix])
    #         emb_con = pd.concat([data, df_emb], axis="columns")
    #     else:
    #         if emb == "struc2vec":
    #             df_emb = catalog.load(emb_files[ix])
    #             df_emb = _preprocess_struc(df_emb)
    #         else:
    #             df_emb = catalog.load(emb_files[ix])
    #         emb_con = pd.concat([emb_con, df_emb], axis="columns")

    # return emb_con


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
