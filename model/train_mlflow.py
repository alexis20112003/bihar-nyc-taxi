import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

import mlflow

from model.load_data import load_train_data, load_test_data
from model.train import preprocess_data, transform_target, NUM_FEATURES, CAT_FEATURES

import common

DIR_MLRUNS = common.get_full_path(common.CONFIG['mlflow']['mlruns'])
EXPERIMENT_NAME = common.CONFIG['mlflow']['experiment_name']
MODEL_NAME = common.CONFIG['mlflow']['model_name']
ARTIFACT_PATH = common.CONFIG['mlflow']['artifact_path']
RANDOM_STATE = int(common.CONFIG['ml']['random_state'])


def build_pipeline(regressor):
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ('scaling', StandardScaler(), NUM_FEATURES)
    ])
    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', regressor)
    ])
    return pipeline


def train_and_log_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    signature = mlflow.models.infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=ARTIFACT_PATH,
        signature=signature)

    return rmse, r2


SAMPLE_SIZE = 50000  # subsample to avoid memory issues

if __name__ == "__main__":

    mlflow.set_tracking_uri("file:" + DIR_MLRUNS)

    # load and preprocess data
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()
    y_train = transform_target(y_train)
    y_test = transform_target(y_test)
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    # subsample to keep it fast and avoid crashes
    if len(X_train) > SAMPLE_SIZE:
        idx = X_train.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).index
        X_train = X_train.loc[idx]
        y_train = y_train.loc[idx]
    if len(X_test) > SAMPLE_SIZE:
        idx = X_test.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).index
        X_test = X_test.loc[idx]
        y_test = y_test.loc[idx]
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_score = float('inf')
    best_run_id = None
    k = 0

    # --- Ridge grid search ---
    params_alpha = [0.1, 1, 10]

    with mlflow.start_run(run_name="ridge", description="Ridge regression grid search") as parent_run:
        for alpha in params_alpha:
            k += 1
            child_run_name = f"ridge_{k:02}"
            pipeline = build_pipeline(Ridge(alpha=alpha))
            with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                mlflow.log_param("algorithm", "Ridge")
                mlflow.log_param("alpha", alpha)
                rmse, r2 = train_and_log_model(pipeline, X_train, X_test, y_train, y_test)
                print(f"[{child_run_name}] alpha={alpha} | RMSE={rmse:.4f} | R2={r2:.4f}")
                if rmse < best_score:
                    best_score = rmse
                    best_run_id = child_run.info.run_id

    # --- RandomForest grid search ---
    params_n_estimators = [50, 100]
    params_max_depth = [5, 10]

    with mlflow.start_run(run_name="randomforest", description="RandomForest grid search") as parent_run:
        for n_estimators in params_n_estimators:
            for max_depth in params_max_depth:
                k += 1
                child_run_name = f"rf_{k:02}"
                regressor = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
                pipeline = build_pipeline(regressor)
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    mlflow.log_param("algorithm", "RandomForest")
                    mlflow.log_param("n_estimators", n_estimators)
                    mlflow.log_param("max_depth", max_depth)
                    rmse, r2 = train_and_log_model(pipeline, X_train, X_test, y_train, y_test)
                    print(f"[{child_run_name}] n_estimators={n_estimators} max_depth={max_depth} | RMSE={rmse:.4f} | R2={r2:.4f}")
                    if rmse < best_score:
                        best_score = rmse
                        best_run_id = child_run.info.run_id

    # Register best model
    print("#" * 40)
    print(f"Best RMSE: {best_score:.4f} (run_id: {best_run_id})")
    model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Model registered: {mv.name} v{mv.version}")
