# Simple MLflow project

This project demonstrates an end-to-end workflow using MLflow to track experiments, save models, and retrieve them via the MLflow API. The workflow consists of data preprocessing, training an ElasticNet model, and loading the trained model for testing.

The data used in this repo comes from [ics.uci.edu](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)

## Project Structure

- `data/wine_quality.csv`: wine quality dataset
- `config.yml`: configuration file (MLflow backend path, data path, etc.)
- `common.py`: script for getting the processed config constants
- `preprocess_data.py`: script for data preprocessing and saving its results into a pickle file: (X_train, X_test, y_train, y_test)
- `train_elasticnet.py`: script for model training and logging experiments conducted using ElasticNet algorithm 
- `test_model_load.py`: script for loading latest model from mlflow
- `requirements.txt`: package requirements file (list of dependencies)


## Run Demo Locally

### Preprocess data
```shell
$ python preprocess_data.py
```
Paths are configured in `config.yml`. Results are saved to `data/` folder accordingly to `config.yml`.

### Model Training & Logging
Train an ElasticNet model and track experiments in MLflow:
```shell
$ python train_elasticnet.py
$ mlflow ui
```
This script logs metrics, parameters, and the trained model in MLflow.
Open http://localhost:5000 to view the mlflow UI.

![Alt text](img/mlflow_experiments_table.png "MLflow experiments table")

### Load the model from mlflow backend
Retrieve the trained model using MLflow API:
```shell
$ python test_model_load.py
```

Here are the expected test results:
```shell
      fixed acidity  volatile acidity  citric acid  residual sugar  ...  sulphates    alcohol  target-true  target-pred
4656            6.0              0.29         0.41            10.8  ...       0.59  10.966667            7     6.306061
3659            5.4              0.53         0.16             2.7  ...       0.53  13.200000            8     6.590940
907             7.1              0.25         0.39             2.1  ...       0.43  12.200000            8     6.364277
4352            7.3              0.28         0.35             1.6  ...       0.47  10.700000            5     5.751044
3271            6.5              0.32         0.34             5.7  ...       0.60  12.000000            7     6.415782

[5 rows x 13 columns]
```