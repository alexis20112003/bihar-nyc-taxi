from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
import pandas as pd
import pickle
import dill

# # need to import DiabetesModel from outside the api root folder
# import os, sys
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.normpath(os.path.join(ROOT_DIR, ".."))
# sys.path.insert(0, PARENT_DIR)
# from model.DiabetesModel import DiabetesModel


import config
DB_PATH = config.CONFIG['paths']['db_path']
MODEL_PATH = config.CONFIG['paths']['model_path']
MODEL_CUSTOM_PATH = config.CONFIG['paths']['model_custom_path']


app = FastAPI()

# load model
print(f"Loading the model from {MODEL_PATH}")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# load model created with a custom wrapper class, including custom preprocessing and postprocessing logic
print(f"Loading the model from {MODEL_CUSTOM_PATH}")
with open(MODEL_CUSTOM_PATH, "rb") as file:
    model_custom = dill.load(file)


class Patient(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    tc: float
    ldl: float
    hdl: float
    tch: float
    ltg: float
    glu: float


@app.post("/predict")
def predict(patient: Patient):

    # get prediction
    input_data = pd.DataFrame([patient.model_dump()])
    # # same as:
    # input_data = np.array([
    #     patient.age,
    #     patient.sex,
    #     patient.bmi,
    #     patient.bp,
    #     patient.tc,
    #     patient.ldl,
    #     patient.hdl,
    #     patient.tch,
    #     patient.ltg,
    #     patient.glu
    # ]).reshape(1, -1)

    result = model.predict(input_data)[0]
    # return prediction
    return {"result": result}

@app.post("/predict_custom")
def predict_custom(patient: Patient):

    # get prediction
    input_data = pd.DataFrame([patient.model_dump()])
    result = model_custom.predict(input_data)[0]

    # return prediction
    return {"result": result}

@app.get("/patients/randomtest")
def get_random_test_patient():
    print(f"Reading random test data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    data_test = pd.read_sql('SELECT * FROM test ORDER BY RANDOM() LIMIT 1', con)
    con.close()
    X = data_test.drop(columns=['target'])
    y = data_test['target']

    return {"x": X.iloc[0], "y": y[0]}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
