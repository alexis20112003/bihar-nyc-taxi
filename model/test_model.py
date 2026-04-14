import numpy as np
import pandas as pd
from model.train import preprocess_data, transform_target
import pickle

from model.load_data import load_random_test_data

import common
MODEL_PATH = common.CONFIG['paths']['model_path']


def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model


def test_model(model):
    print(f"Test the model")
    X, y = load_random_test_data()
    # need to perform the same preprocessing as for training
    X_preprocessed = preprocess_data(X)
    # model predicts log(trip_duration), inverse transform to get seconds
    y_pred_log = model.predict(X_preprocessed)
    y_pred = np.expm1(y_pred_log)
    df = pd.DataFrame()
    df['y_true'] = y.values
    df['y_pred'] = np.round(y_pred).astype(int)
    print(df)


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    test_model(model)
