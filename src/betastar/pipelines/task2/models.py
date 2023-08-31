import pandas as pd

from typing import Any, Dict

import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
):
    model = LogisticRegression(**parameters)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
):
    model = RandomForestClassifier(**parameters)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> lgb.basic.Booster:
    X_tr, X_valid, y_tr, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=2023, stratify=y_train
    )

    train_set = lgb.Dataset(X_tr, label=y_tr)
    valid_set = lgb.Dataset(X_valid, label=y_valid)
    model = lgb.train(
        params=parameters,
        train_set=train_set,
        valid_sets=valid_set,
        # callbacks=[
        #     lgb.reset_parameter(
        #         learning_rate=partial(
        #             _learning_rate_decay,
        #             base_learning_rate=parameters["learning_rate"],
        #             decay_factor=0.99,
        #         )
        #     )
        # ],
    )
    return model


# def _learning_rate_decay(
#     current_interation,
#     base_learning_rate: Optional[float] = 0.1,
#     decay_factor: Optional[float] = 0.99,
# ) -> float:
#     lr = base_learning_rate * np.power(decay_factor, current_interation)
#     return lr if lr > 1e-4 else 1e-4


def train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> xgb.Booster:
    X_tr, X_valid, y_tr, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=2023, stratify=y_train
    )
    model = xgb.XGBClassifier(**parameters)
    # es = xgb.callback.EarlyStopping(
    #     rounds=parameters["early_stopping_rounds"],
    #     save_best=True,
    #     data_name="validation_1",
    #     metric_name=parameters["eval_metric"],
    # )

    eval_set = [(X_tr, y_tr.values.ravel()), (X_valid, y_valid.values.ravel())]

    model.fit(
        X_tr,
        y_tr,
        eval_set=eval_set,
    )  # callbacks=[es])

    return model
