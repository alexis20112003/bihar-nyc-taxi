import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import mlflow, pickle

import common as common

DATA_PROC_PATH = common.CONFIG['paths']['data_processed']
DIR_MLRUNS = common.CONFIG['paths']['mlruns']

RANDOM_STATE = common.CONFIG['ml']['random_state']

EXPERIMENT_NAME = common.CONFIG['mlflow']['experiment_name']
MODEL_NAME = common.CONFIG['mlflow']['model_name']
ARTIFACT_PATH = common.CONFIG['mlflow']['artifact_path']

def load_data():
    with open(DATA_PROC_PATH, "rb") as file:
        X_train, X_test, y_train, y_test = pickle.load(file)
    return X_train, X_test, y_train, y_test

def train_and_log_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    signature = mlflow.models.infer_signature(X_train, y_train)

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=ARTIFACT_PATH,
        signature=signature)

    results = mlflow.evaluate(
        model_info.model_uri,
        data=pd.concat([X_test, y_test], axis=1),
        targets=y_test.name,
        model_type="regressor",
        evaluators=["default"]
    )
    return results

if __name__ == "__main__":

    mlflow.set_tracking_uri("file:" + DIR_MLRUNS)

    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(EXPERIMENT_NAME)

    params_n_estimators = [50, 100, 200]
    params_max_depth = [5, 10, None]

    num_iterations = len(params_n_estimators) * len(params_max_depth)

    run_name = "randomforest"
    k = 0
    best_score = float('inf')
    best_run_id = None

    with mlflow.start_run(run_name=run_name, description=run_name) as parent_run:
        for n_estimators in params_n_estimators:
            for max_depth in params_max_depth:
                k += 1
                print(f"\n***** ITERATION {k} from {num_iterations} *****")
                child_run_name = f"{run_name}_{k:02}"
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE
                )
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    results = train_and_log_model(model, X_train, X_test, y_train, y_test)
                    mlflow.log_param("n_estimators", n_estimators)
                    mlflow.log_param("max_depth", max_depth)
                    rmse = results.metrics['root_mean_squared_error']
                    r2 = results.metrics['r2_score']
                    if rmse < best_score:
                        best_score = rmse
                        best_run_id = child_run.info.run_id
                    print(f"rmse: {rmse}")
                    print(f"r2: {r2}")

    print("#" * 20)
    print(f"Best RandomForest RMSE: {best_score:.4f}")

    # Compare with the best registered ElasticNet model
    mlflow_client = mlflow.MlflowClient()
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest = versions[0]
        elasticnet_run = mlflow_client.get_run(latest.run_id)
        elasticnet_rmse = elasticnet_run.data.metrics.get('root_mean_squared_error')
        print(f"ElasticNet RMSE (registry v{latest.version}): {elasticnet_rmse:.4f}")

        if best_score < elasticnet_rmse:
            print("RandomForest is better! Registering to model registry...")
            model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
            mv = mlflow.register_model(model_uri, MODEL_NAME)
            print(f"Name: {mv.name}, Version: {mv.version}")
        else:
            print("ElasticNet is still better. Not registering RandomForest.")
    else:
        print("No existing model in registry. Registering RandomForest...")
        model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Name: {mv.name}, Version: {mv.version}")
