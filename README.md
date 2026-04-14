# Bihar Taxi

Prediction of NYC taxi trip duration using Ridge/Lasso regression. Separation of data loading, feature engineering, model training, and inference logic.

Based on the [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) competition dataset (~1.4M trips).

## Dataset

| Field | Description |
|-------|-------------|
| `vendor_id` | Provider associated with the trip record |
| `pickup_datetime` | Date and time when the meter was engaged |
| `passenger_count` | Number of passengers in the vehicle |
| `pickup_longitude` / `pickup_latitude` | Pickup coordinates |
| `dropoff_longitude` / `dropoff_latitude` | Dropoff coordinates |
| `store_and_fwd_flag` | Whether the trip record was held in vehicle memory (Y/N) |
| `trip_duration` | Duration of the trip in seconds (**target variable**) |

## Project Structure

```
.
├── config.yml              # Paths and ML configuration
├── common.py               # Config loader utility
├── requirements.txt        # Python dependencies
├── data/
│   └── download_data.py    # Download and store data in SQLite (data/taxi.db)
├── notebooks/
│   └── LAB_5_Regression_Linear_Models_SOLUTION.ipynb  # EDA and model exploration
├── model/
│   ├── load_data.py        # Data access layer (train/test from SQLite)
│   ├── train.py            # Train basic model and save to model-registry/
│   ├── train_custom_model.py  # Train custom wrapper model
│   └── test_model.py       # Test inference with pretrained model
└── api/
    └── main.py             # FastAPI prediction endpoints
```

## Setup

### 1. Create and activate a virtual environment
```shell
$ python -m venv venv
$ source venv/bin/activate
```

### 2. Install dependencies
```shell
$ pip install -r requirements.txt
```

## Run the Training Pipeline

### 1. Load data into the database
```shell
$ python -m data.download_data
```
Creates `data/taxi.db`, a SQLite database containing 2 tables: `train` and `test`.

### 2. Train and save the model
```shell
$ python -m model.train
```
Creates `model-registry/taxi.model`, a serialized Ridge regression pipeline.

### 3. Test inference
```shell
$ python -m model.test_model
```
Loads the saved model and evaluates on random test samples.

## Custom Wrapper Model

The `model.TaxiModel` class wraps a sklearn pipeline with custom preprocessing (feature engineering from `pickup_datetime`, haversine distance, traffic/speed flags) and postprocessing (inverse log transform).

```shell
$ python -m model.train_custom_model
```
Creates `model-registry/taxi_custom.model`.

## Taxi Trip Duration Prediction API

### Overview

FastAPI-based API for predicting taxi trip duration. Paths configured in `config.yml`.

### Endpoints

#### POST /predict

Predicts trip duration using the primary model `model-registry/taxi.model`.

Request Body example (JSON):
```json
{
    "vendor_id": 1,
    "pickup_datetime": "2016-03-14 17:24:55",
    "passenger_count": 1,
    "pickup_longitude": -73.9821,
    "pickup_latitude": 40.7678,
    "dropoff_longitude": -73.9645,
    "dropoff_latitude": 40.7654,
    "store_and_fwd_flag": "N"
}
```

Response:
```json
{
  "result": 643
}
```

#### POST /predict_custom

Predicts trip duration using the custom model `model-registry/taxi_custom.model`.

#### GET /patients/randomtest

Retrieves a random test trip from the database.
