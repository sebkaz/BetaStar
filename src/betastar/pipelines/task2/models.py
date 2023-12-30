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

#### GNN code

import torch
from torch.nn import Linear
import torch.nn.functional as F

def accuracy(y_pred, y_true):
    return torch.sum(y_pred==y_true)/len(y_true)

class MLP(torch.nn.Module):

    def __init__(self,dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1),data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f}  | Train Acc: {acc*100:>5.2f}% | Val Loss:{val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
        
    def test(self,data):
        self.eval()
        out = self(data.x)
        return accuracy(out.argmax(dim=1)[data.test_mask],data.y[data.test_mask])
    
    



