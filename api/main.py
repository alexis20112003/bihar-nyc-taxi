from fastapi import FastAPI
from pydantic import BaseModel, model_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated
from validators import check_longitude, check_latitude, check_haversine_distance
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

from service import save_prediction


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
    pickup_longitude: Annotated[float, AfterValidator(check_longitude)]
    pickup_latitude: Annotated[float, AfterValidator(check_latitude)]
    dropoff_longitude: Annotated[float, AfterValidator(check_longitude)]
    dropoff_latitude: Annotated[float, AfterValidator(check_latitude)]
    store_and_fwd_flag: str

    @model_validator(mode='after')
    def validate_distance(self):
        check_haversine_distance(self)
        return self


@app.post("/predict")
def predict(trip: Trip):

    # get prediction
    input_data = pd.DataFrame([trip.model_dump()])
    input_preprocessed = preprocess_data(input_data)
    # model predicts log(trip_duration), inverse transform to get seconds
    result_log = model.predict(input_preprocessed)[0]
    result = int(np.round(np.expm1(result_log)))
    # persist prediction
    save_prediction(trip.model_dump(), result, "predict")
    # return prediction
    return {"result": result}

@app.post("/predict_custom")
def predict_custom(trip: Trip):

    # get prediction (TaxiModel handles preprocessing and postprocessing)
    input_data = pd.DataFrame([trip.model_dump()])
    result = int(model_custom.predict(input_data)[0])
    # persist prediction
    save_prediction(trip.model_dump(), result, "predict_custom")
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
