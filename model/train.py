import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

import os
import pickle

from model.load_data import load_train_data, load_test_data

import common
MODEL_PATH = common.CONFIG['paths']['model_path']

# abnormal dates detected in the notebook EDA (fewer than 6300 trips)
ABNORMAL_DATES = ['2016-01-23', '2016-01-24', '2016-01-25', '2016-05-30']

# feature lists for the pipeline
NUM_FEATURES = ['log_distance_haversine', 'hour',
                'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                'is_rare_pickup_point', 'is_rare_dropoff_point']
CAT_FEATURES = ['weekday', 'month']
TRAIN_FEATURES = NUM_FEATURES + CAT_FEATURES


# --- helper functions (from notebook) ---

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def is_high_traffic_trip(X):
    return ((X['hour'] >= 8) & (X['hour'] <= 19) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
           ((X['hour'] >= 13) & (X['hour'] <= 20) & (X['weekday'] == 5))


def is_high_speed_trip(X):
    return ((X['hour'] >= 2) & (X['hour'] <= 5) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
           ((X['hour'] >= 4) & (X['hour'] <= 7) & (X['weekday'] >= 5) & (X['weekday'] <= 6))


def is_rare_point(X, latitude_column, longitude_column, qmin_lat, qmax_lat, qmin_lon, qmax_lon):
    lat_min = X[latitude_column].quantile(qmin_lat)
    lat_max = X[latitude_column].quantile(qmax_lat)
    lon_min = X[longitude_column].quantile(qmin_lon)
    lon_max = X[longitude_column].quantile(qmax_lon)
    res = (X[latitude_column] < lat_min) | (X[latitude_column] > lat_max) | \
          (X[longitude_column] < lon_min) | (X[longitude_column] > lon_max)
    return res


# --- preprocessing ---

def transform_target(y):
    return np.log1p(y)


def preprocess_data(X):
    print(f"Preprocessing data")
    res = X.copy()

    # ensure pickup_datetime is datetime (stored as text in SQLite)
    res['pickup_datetime'] = pd.to_datetime(res['pickup_datetime'])

    # step 1: datetime features
    res['weekday'] = res['pickup_datetime'].dt.weekday
    res['month'] = res['pickup_datetime'].dt.month
    res['hour'] = res['pickup_datetime'].dt.hour
    abnormal_dates = pd.to_datetime(ABNORMAL_DATES).date
    res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates).astype(int)

    # step 2: distance and traffic features
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

    # transform target
    y_train = transform_target(y_train)

    # preprocess features
    X_train_preprocessed = preprocess_data(X_train)

    # build and fit pipeline
    model = build_pipeline()
    model.fit(X_train_preprocessed, y_train)

    # Evaluate the model on train data
    y_pred = model.predict(X_train_preprocessed)
    score = root_mean_squared_error(y_train, y_pred)
    print(f"Train RMSE = {score:.4f}")

    return model


def evaluate_model(model):
    print(f"Evaluating the model")

    # load test data
    X_test, y_test = load_test_data()

    # transform target
    y_test = transform_target(y_test)

    # need to do the same preprocessing as for train data
    X_test_preprocessed = preprocess_data(X_test)
    y_pred = model.predict(X_test_preprocessed)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE = {rmse:.4f}")
    print(f"Test R2 = {r2:.4f}")

    return rmse


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
