from tokenize import Double
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
# import xgboost as xgb

class Input(BaseModel):
    police_station: str
    light: str
    weather: str
    hit_and_run: str 
    highway: str 
    type_of_collision: str
    zone: str
    lat: float 
    lon: float
    year: float
    month: float 
    day: float 
    date: float
    hour: float
    temp: float
    dwpt: float 
    rhum: float 
    prcp: float 
    wdir: float 
    wspd: float
    pres: float 
    coco: float 
    diff_in_days: float
    shape_length: float 
    lon_factor: float
    lat_factor: float

app = FastAPI()

@app.post("/predict/")
async def predict(input: Input):
    # Preprocessing of Input - remember to import required libraries and add in Procfile
    # Depickling of model and Prediction
    # loaded_model = pickle.load(open('XGB_Best_Model.pkl', 'rb')) 
    return input # Return prediction etc. whatever UI wants