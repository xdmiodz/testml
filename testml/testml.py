import pandas as pd
import sklearn
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import logging
import pickle

def convert_labels_to_ints(labels):
    return (labels == "f").astype(int)

def convert_ints_to_labels(ints):
    return ints.replace(to_replace={0: "t", 1: "f"})

def load_data(data_path):
    data = pd.read_csv(data_path, delimiter=";", header=None)

    # give name to the columns for better debugging
    feature_column_names = {idx + 2: "f{}".format(idx) for idx in range(len(data.columns) - 2)}
    columns_names = {0: "key", 1: "group"}
    columns_names.update(feature_column_names)
    data.rename(columns=columns_names, inplace = True)

    X = data[list(feature_column_names.values()) + ["key",]]
    y = data["group"]

    categorical_features = ["f3",]
    X = pd.get_dummies(X, columns=categorical_features)

    return X, y

def train_model(X_train, y_train):
    y_train = convert_labels_to_ints(y_train)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    X_train = X_train.drop(columns=["key",])

    xgbclf = XGBClassifier(nthread=-1, n_estimators=100, scale_pos_weight=scale_pos_weight)
    xgbclf.fit(X_train, y_train)

    return xgbclf

def test_model(xgbclf, X_test, y_test):
    X_test = X_test.drop(columns=["key",])
    y_test = convert_labels_to_ints(y_test)

    y_pred_proba = xgbclf.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_true=y_test, y_score=y_pred_proba)
    return score

def eval_model(xgbclf, X_eval):
    key = X_eval[["key",]]
    X_eval = X_eval.drop(columns=["key",])

    y_eval = xgbclf.predict(X_eval)
    y_eval = pd.DataFrame(y_eval, columns=["group",])
    y_eval = convert_ints_to_labels(y_eval)

    return pd.concat([key, y_eval], axis=1)

def store_model(xgbclf, path):
    with open(path, "wb") as storage:
        pickle.dump(xgbclf, storage)

def load_model(path):
    with open(path, "rb") as storage:
        xgbclf = pickle.load(storage)
    return xgbclf


def do_train(data_path, model_path):
    logging.info("loading data from {}".format(data_path))
    X_train, y_train = load_data(data_path)

    logging.info("training model")
    xgbclf = train_model(X_train, y_train)

    logging.info("storing model")
    store_model(xgbclf, model_path)

def do_test(data_path, model_path):
    logging.info("loading data from {}".format(data_path))
    X_test, y_test = load_data(data_path)

    logging.info("loading model from {}".format(model_path))
    xgbclf = load_model(model_path)

    score = test_model(xgbclf, X_test, y_test)
    logging.info("ROC AUC score: {:.5f}".format(score))


def do_eval(data_path, model_path, eval_path):
    logging.info("loading data from {}".format(data_path))
    X_eval, _ = load_data(data_path)

    logging.info("loading model from {}".format(model_path))
    xgbclf = load_model(model_path)

    logging.info("evaluating model")
    y_eval = eval_model(xgbclf, X_eval)

    if eval_path is not None:
        y_eval.to_csv(eval_path, index=False, header=None, sep=";")
    else:
        print(y_eval)

