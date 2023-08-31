from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any, Dict


def train_linear_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
):
    model = LinearRegression(**parameters)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
):
    model = Ridge(**parameters)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
):
    model = RandomForestRegressor(**parameters)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> xgb.Booster:
    X_tr, X_valid, y_tr, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=2023
    )
    model = xgb.XGBRegressor(**parameters)
    eval_set = [(X_tr, y_tr.values.ravel()), (X_valid, y_valid.values.ravel())]
    model.fit(X_tr, y_tr, eval_set=eval_set)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> lgb.Booster:
    X_tr, X_valid, y_tr, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=2023
    )
    train_set = lgb.Dataset(X_tr, label=y_tr)
    valid_set = lgb.Dataset(X_valid, label=y_valid)
    model = lgb.train(
        params=parameters,
        train_set=train_set,
        valid_sets=valid_set,
    )

    return model
