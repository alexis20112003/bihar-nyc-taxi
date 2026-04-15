from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
import numpy as np
import pandas as pd
import pickle
import dill
import os
import sys

# need to import model helpers from outside the api root folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.normpath(os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)
from model.train import preprocess_data

import config
DB_PATH = config.CONFIG['paths']['db_path']
MODEL_PATH = config.CONFIG['paths']['model_path']
MODEL_CUSTOM_PATH = config.CONFIG['paths']['model_custom_path']


app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

# load model (Ridge pipeline trained on log-transformed target)
print(f"Loading the model from {MODEL_PATH}")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# load model created with a custom wrapper class, including custom preprocessing and postprocessing logic
print(f"Loading the model from {MODEL_CUSTOM_PATH}")
with open(MODEL_CUSTOM_PATH, "rb") as file:
    model_custom = dill.load(file)


class Trip(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str


@app.post("/predict")
def predict(trip: Trip):

    # get prediction
    input_data = pd.DataFrame([trip.model_dump()])
    input_preprocessed = preprocess_data(input_data)
    # model predicts log(trip_duration), inverse transform to get seconds
    result_log = model.predict(input_preprocessed)[0]
    result = int(np.round(np.expm1(result_log)))
    # return prediction
    return {"result": result}

@app.post("/predict_custom")
def predict_custom(trip: Trip):

    # get prediction (TaxiModel handles preprocessing and postprocessing)
    input_data = pd.DataFrame([trip.model_dump()])
    result = int(model_custom.predict(input_data)[0])

    # return prediction
    return {"result": result}

@app.get("/trips/randomtest")
def get_random_test_trip():
    print(f"Reading random test data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    data_test = pd.read_sql('SELECT * FROM test ORDER BY RANDOM() LIMIT 1', con)
    con.close()
    X = data_test.drop(columns=['trip_duration'])
    y = data_test['trip_duration']

    return {"x": X.iloc[0].to_dict(), "y": int(y.iloc[0])}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
