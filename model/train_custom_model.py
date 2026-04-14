import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import os
import dill

from model.load_data import load_train_data, load_test_data

import common
MODEL_PATH = common.CONFIG['paths']['model_custom_path']


# Custom wrapper class for Diabetes prediction model
# It includes custom preprocessing and postprocessing logic
class DiabetesModel:
    def __init__(self, model):
        self.model = model

    def __preprocess(X):
        # your features preprocessing logic: adding new features, etc.
        # example: X['weekday'] = X['pickup_datetime'].dt.weekday
        return X

    def __preprocess_target(self, y):
        # your target preprocessing logic
        # example: y = np.log1p(y)
        return y

    def __postprocess_target(self, raw_output):
        # your target postprocessing logic: inverse transformation
        # example: output = np.expm1(raw_output)
        output = np.round(raw_output)
        return output

    def fit(self, X, y):
        X_processed = self.__preprocess(X)
        y_processed = self.__preprocess_target(y)
        self.model.fit(X_processed, y_processed)
        return self

    def predict(self, X):
        try:
            check_is_fitted(self.model)
            X_processed = self.__preprocess(X)
            raw_output = self.model.predict(X_processed)
            output = self.__postprocess_target(raw_output)
        except NotFittedError as exc:
            print(f"Model is not fitted yet.")
        return output


def train_model():
    print(f"Building a model")

    # load train data
    X_train, y_train = load_train_data()

    # Build and wrap a model
    model = LinearRegression()
    model_wrapped = DiabetesModel(model)
    model_wrapped.fit(X_train, y_train)

    # Evaluate the model on train data
    y_pred = model_wrapped.predict(X_train)
    score = mean_squared_error(y_train, y_pred)
    print(f"Score on train data {score:.2f}")

    return model_wrapped


def evaluate_model():
    print(f"Evaluating the model")

    # load test data
    X_test, y_test = load_test_data()

    # # no need to do preprocessing, since it is already encapsulated in DiabetesModel::predict
    # X_preprocessed = preprocess_data(X)
    # y_pred = model.predict(X_preprocessed)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    print(f"Score on test data {score:.2f}")

    return score


def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # When you use dill, in comparison with pickle,
    # you don't need to explicitly import the class when deserializing.
    with open(path, "wb") as file:
        dill.settings['recurse'] = True
        dill.dump(model, file)
    print(f"Done")


if __name__ == "__main__":

    # training workflow
    # fit model
    model = train_model()
    # evaluate model
    score = evaluate_model()
    # serialize model in a file
    persist_model(model, MODEL_PATH)
