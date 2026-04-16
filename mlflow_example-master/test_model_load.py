import pandas as pd
import mlflow
import pickle
import common as common

DATA_PROC_PATH = common.CONFIG['paths']['data_processed']
DIR_MLRUNS = common.CONFIG['paths']['mlruns']

MODEL_NAME = common.CONFIG['mlflow']['model_name']

def load_data():

    with open(DATA_PROC_PATH, "rb") as file:
        _, X_test, _, y_test = pickle.load(file)
    return X_test, y_test

if __name__ == "__main__":

    # Load preprocessed test data
    X_test, y_test = load_data()
    X_test = X_test[:5]
    y_test = y_test[:5]

    # Using MLflow Tracking locally: everything will be stored in DIR_MLRUNS folder
    mlflow.set_tracking_uri("file:" + DIR_MLRUNS)

    # This is to show how to load the latest version of a model
    # from the mlflow model registry

    mlflow_client = mlflow.MlflowClient()
    model_metadata = mlflow_client.get_latest_versions(MODEL_NAME, stages=["None"])
    latest_model_version = model_metadata[0].version

    print("Load model from the model registry")
    model_uri = f"models:/{MODEL_NAME}/{latest_model_version}"
    print(f"Model URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    y_pred = model.predict(X_test[:5])

    data_test = X_test
    data_test['target-true'] = y_test
    data_test['target-pred'] = y_pred
    print(data_test)

