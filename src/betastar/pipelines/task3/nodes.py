"""
This is a boilerplate pipeline 'feature_importance'
generated using Kedro 0.18.11
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

COMMUNITY_LIST = ["cada", "edge_contr", "in_mod_deg", "participation", "beta_star", 
    "l1", "l2", "kl", "hd", "l12", "l22", "kl2", "hd2",]

NOCOMMUNITY_LIST = ["lcc", "bc", "cc", "dc", "ndc", "ec", "eccen", "core"]

EMB_PREFIX = ("emb_")
METRICS = [roc_auc_score, average_precision_score]



def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
):
    model = RandomForestClassifier(**parameters)
    model.fit(X_train, y_train.values.ravel())
    return model

def pi_forest(X, y, forest):
    premu_rfr_train = permutation_importance(
        forest, X, y, n_repeats=10, random_state=2023, scoring="average_precision"
    )
    names = ["Random Forest"]
    results = [premu_rfr_train]
    graph_data = {}
    for result, name in zip(results, names):
        graph_data[name] = result["importances_mean"]
    graph_data = pd.DataFrame.from_dict(graph_data, orient="index", columns=X.columns)
    graph_data.reset_index(inplace=True, drop=False)
    graph_data.rename(columns={"index": "model_name"}, inplace=True)
    graph_data = graph_data.melt(id_vars="model_name")
    return graph_data