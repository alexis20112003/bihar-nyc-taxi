# Diabetes

Example of separation of data loading and processing, model training and inference logic.

## Files for the basic Training Pipeline

- `data/download_data.py`: Script for loading data into a local SQLite database (`data\diabetes.db`)
- `notebooks/EDA.ipynb`: Jupyter notebook for basic exploratory data analysis (EDA).
- `model/load_data.py`: Script for loading train and test data (data access layer)
- `model/train.py`: Script for training a model and saving it to the `model-registry` folder
- `model/test_model.py`: Script for testing the access to the pretrained model.
- `common.py`: Script for loading and preparing config constants from `config.yml` file.
- `requirements.txt`: List of required Python packages.

## Run the basic Training Pipeline

### 1. Load data into the database
Run the following command to download and store the dataset in an SQLite database:
```shell
$ python -m data.download_data
```
This creates the `data/diabetes.db` file, a [lightweight disk-based database](https://docs.python.org/3/library/sqlite3.html) containing 2 tables: train and test.

### 2. Train and save the model
To train the model and save it, run:
```shell
$ python -m model.train
```
This creates the `models/diabetes.model` file, which contains a serialized regression model.

### 3. Test inference
To test the trained model using the test dataset, run:
```shell
python -m model.test_model
```
This script loads the previously saved model and evaluates its performance.

## Run the Training Pipeline using Custom Wrapper Class

The `model.DiabetesModel` class is a wrapper around a basic machine learning model, providing preprocessing and postprocessing capabilities.

It includes standard `fit(X, y)` and `predict(X)` methods and allows to include specific feature engineering and output transformations relevant to your use case by customizing `_preprocess` and `_postprocess` methods.

Train and save the model based on `model.DiabetesModel` class:
```shell
$ python -m model.train_custom_model
```
This creates the `models/diabetes_custom.model` file, which contains a serialized regression model.

## Diabetes Prediction API

### Overview

This FastAPI-based API provides endpoints for predicting diabetes progression based on patient health metrics using a trained machine learning model. Model and database paths are defined in `config.yml`.

### Endpoints

#### POST /predict

Predicts diabetes progression using the primary model `models/diabetes.model`.

Request Body example (JSON):
```
{
    "age": 0.09619652164973376,
    "sex": -0.044641636506989144,
    "bmi": 0.05199589785375607,
    "bp": 0.0792647112814439,
    "tc": 0.05484510736603471,
    "ldl": 0.036577086450315016,
    "hdl": -0.07653558588880739,
    "tch": 0.14132210941786577,
    "ltg": 0.0986480615153178,
    "glu": 0.06105390622205087
}
```

Response:
```
{
  "result": 291.4170292522082
}
```

#### POST /predict_custom

Predicts diabetes progression using the custom model `models/diabetes_custom.model`.

Response for the same request example as for `POST /predict` above (rounded to integer):
```
{
  "result": 291
}
```

#### GET /patients/randomtest

Retrieves a random test patient from the database.
