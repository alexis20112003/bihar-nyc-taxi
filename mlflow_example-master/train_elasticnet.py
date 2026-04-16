import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet

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
    # y_pred = model.predict(X_test)

    # Infer an MLflow model signature from the training data (input),
    # model predictions (output) and parameters (for inference).
    signature = mlflow.models.infer_signature(X_train, y_train)

    # Log model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=ARTIFACT_PATH,
        signature=signature)

    # Log model params
    # mlflow.log_params(model.get_params())

    # Log metrics & artifacts to MLflow tracking server
    results = mlflow.evaluate(
        model_info.model_uri,
        data=pd.concat([X_test,y_test], axis=1),
        targets=y_test.name,
        model_type="regressor",
        evaluators=["default"]
    )
    return results

if __name__ == "__main__":

    # Using MLflow Tracking locally: everything will be stored in DIR_MLRUNS folder
    mlflow.set_tracking_uri("file:" + DIR_MLRUNS)

    # load preprocessed data
    X_train, X_test, y_train, y_test = load_data()

    # set mlflow experiment
    exp_name = "wine_quality_prediction"
    # experiment_id = mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    params_alpha = [0.01, 0.1, 1, 10]
    params_l1_ratio = np.arange(0.0, 1.1, 0.5)
    # params_alpha = [0.5]
    # params_l1_ratio = [0.5]

    num_iterations = len(params_alpha) * len(params_l1_ratio)

    run_name = "elasticnet"
    k = 0
    best_score =float('inf')
    best_run_id = None

    # Test all the defined combinations of hyperparams
    # Log each run
    # Register the best model
    with mlflow.start_run(run_name=run_name, description=run_name) as parent_run:
        for alpha in params_alpha:
            for l1_ratio in params_l1_ratio:
                k += 1
                print(f"\n***** ITERATION {k} from {num_iterations} *****")
                child_run_name = f"{run_name}_{k:02}"
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    results = train_and_log_model(model, X_train, X_test, y_train, y_test)
                    # log hyperparameters
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("l1_ratio", l1_ratio)
                    if results.metrics['root_mean_squared_error'] < best_score:
                        best_score = results.metrics['root_mean_squared_error']
                        best_run_id = child_run.info.run_id
                    print(f"rmse: {results.metrics['root_mean_squared_error']}")
                    print(f"r2: {results.metrics['r2_score']}")

    print("#" * 20)
    # Register the baseline in the model registry
    model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print("Model saved to the model registry:")
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Source: {mv.source}")