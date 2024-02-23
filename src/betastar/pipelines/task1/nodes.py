import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any, Dict
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, kendalltau

from .models import (
    train_linear_model,
    train_ridge_model,
    train_random_forest,
    train_xgb,
    train_lightgbm,
)


logger = logging.getLogger(__name__)

TARGET_LIST = [
    "cada",
    "edge_contr",
    "in_mod_deg",
    "participation",
    "beta_star",
    "l1",
    "l2",
    "kl",
    "hd",
    "l12",
    "l22",
    "kl2",
    "hd2",
]

FEATURES_LIST = ["lcc", "bc", "cc", "dc", "ndc", "ec", "eccen", "core"]

MODELS = [
    ("linear_model", train_linear_model, "lin_reg"),
    ("ridge", train_ridge_model, "ridge"),
    ("random_forest", train_random_forest, "random_forest_reg"),
    ("xgboost", train_xgb, "xgboost_reg"),
    ("lightgbm", train_lightgbm, "lightgbm_reg"),
]

MEASURES = [
    ("r2_score", r2_score),
    ("spearmans", spearmanr),
    ("kendall", kendalltau),
]


def train_test_split_df(data: pd.DataFrame, parameters: Dict[str, Any]):
    """Split our data into train and test sets"""
    X = data.drop(columns=["target", "node_id"], axis=1)

    X_train, X_test = train_test_split(
        X, test_size=parameters["test_size"], random_state=2023
    )
    for dataset in [X_train, X_test]:
        dataset.reset_index(drop=True, inplace=True)

    return dict(
        X_train_linear=X_train,
        X_test_linear=X_test,
    )


def get_results(X_train, X_test, parameters):
    all_features = X_train.columns.tolist()

    if not parameters['grid']['search']:
        emb = [f for f in all_features if f.startswith(parameters["emb_prefix"])]
        emb = [f for f in emb if ("p1q1" in f) or ("struc" in f)]
        features = FEATURES_LIST + emb
        logger.info(f"You run model for p=1 and q=1 with {len(features)} features")
        results, predictions = _one_computation(features, X_train, X_test, parameters)
    else:
        results = pd.DataFrame()
        logger.info("You run models for different p and q")
        for p in parameters['grid']['p']:
            for q in parameters['grid']['q']:

                emb = [f for f in all_features if f.startswith(parameters["emb_prefix"])]
                emb = [f for f in emb if (f"p{p}q{q}" in f) or ("struc" in f)]

                features = FEATURES_LIST + emb
                logger.info(f"You run model for p={p} and q={q} with {len(features)} features")
                result, prediction = _one_computation(features, X_train, X_test, parameters, p=p, q=q)
                results = pd.concat([results, result], axis=0)
                predictions = {f"p{p}q{q}": prediction}
    return dict(results=results, predictions=predictions)


def _one_computation(features, X_train, X_test, parameters, p=1, q=1):
    results = {"target": [], "model_name": [], "measure": [], "value": []}
    predictions = {
        "model_name": [],
        "target": [],
        "y_pred": [],
    }
    for target in TARGET_LIST:
            logger.info(f"Model for target: {target}")

            for model_name, linear_model, pname in MODELS:
                logger.info(f"Model: {model_name} for target: {target} with {len(features)} features ")
                model = linear_model(X_train[features], X_train[target], parameters[pname])
                y_pred = model.predict(X_test[features])
                # predictions["model_name"].append(model_name+"_"+str(p)+"_"+str(q))
                # predictions["target"].append(target)
                # predictions["y_pred"].append(y_pred)

                for metric, func in MEASURES:
                    results["model_name"].append(model_name+"_"+str(p)+"_"+str(q))
                    results["measure"].append(metric)
                    results["target"].append(target)
                    if metric == "r2_score":
                        results["value"].append(func(X_test[target], y_pred))
                    else:
                        results["value"].append(func(X_test[target], y_pred).statistic)
    return pd.DataFrame(results), predictions


def mutual_information(X_train, X_test, parameters):
    pass