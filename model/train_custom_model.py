import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import os
import dill

from model.load_data import load_train_data, load_test_data
from model.train import (
    haversine_array, is_high_traffic_trip, is_high_speed_trip, is_rare_point,
    ABNORMAL_DATES, NUM_FEATURES, CAT_FEATURES, TRAIN_FEATURES
)

import common
MODEL_PATH = common.CONFIG['paths']['model_custom_path']


# Custom wrapper class for Taxi trip duration prediction model
# It includes custom preprocessing and postprocessing logic
class TaxiModel:
    def __init__(self, model):
        self.model = model

    def _preprocess(self, X):
        res = X.copy()
        res['pickup_datetime'] = pd.to_datetime(res['pickup_datetime'])
        res['weekday'] = res['pickup_datetime'].dt.weekday
        res['month'] = res['pickup_datetime'].dt.month
        res['hour'] = res['pickup_datetime'].dt.hour
        abnormal_dates = pd.to_datetime(ABNORMAL_DATES).date
        res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates).astype(int)
        distance_haversine = haversine_array(
            res.pickup_latitude, res.pickup_longitude,
            res.dropoff_latitude, res.dropoff_longitude
        )
        res['log_distance_haversine'] = np.log1p(distance_haversine)
        res['is_high_traffic_trip'] = is_high_traffic_trip(res).astype(int)
        res['is_high_speed_trip'] = is_high_speed_trip(res).astype(int)
        res['is_rare_pickup_point'] = is_rare_point(
            res, "pickup_latitude", "pickup_longitude", 0.01, 0.995, 0, 0.95
        ).astype(int)
        res['is_rare_dropoff_point'] = is_rare_point(
            res, "dropoff_latitude", "dropoff_longitude", 0.01, 0.995, 0.005, 0.95
        ).astype(int)
        return res[TRAIN_FEATURES]

    def _preprocess_target(self, y):
        return np.log1p(y)

    def _postprocess_target(self, raw_output):
        output = np.round(np.expm1(raw_output))
        return output

    def fit(self, X, y):
        X_processed = self._preprocess(X)
        y_processed = self._preprocess_target(y)
        self.model.fit(X_processed, y_processed)
        return self

    def predict(self, X):
        try:
            check_is_fitted(self.model)
            X_processed = self._preprocess(X)
            raw_output = self.model.predict(X_processed)
            output = self._postprocess_target(raw_output)
        except NotFittedError as exc:
            print(f"Model is not fitted yet.")
            output = None
        return output


def build_pipeline():
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ('scaling', StandardScaler(), NUM_FEATURES)
    ])
    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])
    return pipeline


def train_model():
    print(f"Building a model")

    # load train data
    X_train, y_train = load_train_data()

    # Build and wrap a model
    pipeline = build_pipeline()
    model_wrapped = TaxiModel(pipeline)
    model_wrapped.fit(X_train, y_train)

    # Evaluate the model on train data (on log-transformed target)
    y_pred = model_wrapped.predict(X_train)
    score = root_mean_squared_error(y_train, y_pred)
    print(f"Score on train data (RMSE in seconds) {score:.2f}")

    return model_wrapped


def evaluate_model(model):
    print(f"Evaluating the model")

    # load test data
    X_test, y_test = load_test_data()

    # no need to do preprocessing, since it is already encapsulated in TaxiModel::predict
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Score on test data (RMSE in seconds) {rmse:.2f}")

    return rmse


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
    score = evaluate_model(model)
    # serialize model in a file
    persist_model(model, MODEL_PATH)
