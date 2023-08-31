from typing import Any, Dict
import logging
import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn import preprocessing as pp

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from .models import (
    train_logistic_regression,
    train_random_forest_classifier,
    train_lightgbm_classifier,
    train_xgb_classifier,
)

logger = logging.getLogger(__name__)


MODEL_NAMES = ["log_reg", "random_forest", "lightgbm", "xgboost"]

CLASSIFIERS = [
    train_logistic_regression,
    train_random_forest_classifier,
    train_lightgbm_classifier,
    train_xgb_classifier,
]

METRICS = [roc_auc_score, average_precision_score]


def standarisation(data: pd.DataFrame):
    featureToScale = data.drop(["target", "node_id"], axis=1).columns
    ss = pp.StandardScaler(copy=True)
    data.loc[:, featureToScale] = ss.fit_transform(data[featureToScale])
    return data


def split_df(data: pd.DataFrame, parameters: Dict[str, Any]):
    """Split our data into train and test sets"""
    X, y = data.drop(columns=["target", "node_id"], axis=1), data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=2023, stratify=y
    )
    for dataset in [X_train, X_test, y_train, y_test]:
        dataset.reset_index(drop=True, inplace=True)

    return dict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.to_frame(),
        y_test=y_test.to_frame(),
    )


def features_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters: Dict[str, Any],
    params_off: list,
):
    results = {"features": [], "model_name": [], "measure": [], "value": []}
    features = X_train.columns.tolist()  # all features
    our_features = [f for f in features if not f.startswith(parameters["emb_prefix"])]
    # next model checker for choosed features in parameters file
    test_features = parameters["test_features"]
    models_list = our_features + [test_features]
    emb_features_all = [f for f in features if f.startswith(parameters["emb_prefix"])]
    for emb in parameters["embeddings"]:
        models_list += [[f for f in emb_features_all if emb in f]]

    for model_f in models_list:
        if isinstance(model_f, str):
            logger.info(f"Training model with our feature: {model_f}")
            model_features = [model_f]
        else:
            if model_f[0] == test_features[0]:
                model_f = "participation_in_mod_deg"
                model_features = test_features
            else:
                model_features = model_f
                model_f = model_f[0].split("_")[1]

        for name, clf in zip(MODEL_NAMES, CLASSIFIERS):
            try:
                logger.info(f"Training model: {name} ...")
                # create model
                model = clf(X_train[model_features], y_train, parameters[name])
                # models[f"{model_f}_{name}"] = model
                # get predictions
                if isinstance(model, lgb.basic.Booster):
                    y_pred = model.predict(X_test[model_features])
                else:
                    y_pred = model.predict_proba(X_test[model_features])[:, 1]
                    # get metrics
                for metric in METRICS:
                    if metric.__name__ == "roc_auc_score":
                        results["features"].append(model_f)
                        results["model_name"].append(name)
                        results["measure"].append(metric.__name__)
                        results["value"].append(round(metric(y_test, y_pred), 3))
                    else:
                        results["features"].append(model_f)
                        results["model_name"].append(name)
                        results["measure"].append(metric.__name__)
                        results["value"].append(round(metric(y_test, y_pred), 3))
                        for cutoff in params_off:
                            results["features"].append(model_f)
                            results["model_name"].append(name)
                            results["measure"].append(f"{metric.__name__}_{cutoff}")
                            results["value"].append(
                                round(
                                    metric(y_test, np.where(y_pred > cutoff, 1, 0)), 3
                                )
                            )
            except:
                results["features"].append(model_f)
                results["model_name"].append(name)
                results["measure"].append("error")
                results["value"].append(0)

    return pd.DataFrame(results)
