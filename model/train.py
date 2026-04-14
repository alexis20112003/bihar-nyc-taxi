from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import os
import pickle

from model.load_data import load_train_data, load_test_data

import common
MODEL_PATH = common.CONFIG['paths']['model_path']


def preprocess_data(X):
    print(f"Preprocessing data")
    return X


def train_model():
    print(f"Building a model")

    # load train data
    X_train, y_train = load_train_data()

    # Build a model
    model = LinearRegression()
    X_train_preprocessed = preprocess_data(X_train)
    model.fit(X_train_preprocessed, y_train)

    # Evaluate the model on train data
    y_pred = model.predict(X_train_preprocessed)
    score = mean_squared_error(y_train, y_pred)
    print(f"Score on train data {score:.2f}")

    return model


def evaluate_model(model):
    print(f"Evaluating the model")

    # load test data
    X_test, y_test = load_test_data()

    # need to do the same preprocessing as for train data
    X_test_preprocessed = preprocess_data(X_test)
    y_pred = model.predict(X_test_preprocessed)
    score = mean_squared_error(y_test, y_pred)
    print(f"Score on test data {score:.2f}")

    return score


def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")


if __name__ == "__main__":

    # training workflow
    # fit model
    model = train_model()
    # evaluate model
    score = evaluate_model(model)
    # serialize model in a file
    persist_model(model, MODEL_PATH)
