from tokenize import Double
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import json

def convert_input_list_to_json_input(custom_input_list):
    d = dict()
    keys = [
        'light',
        'weather',
        'highway',
        'type_of_collision',
        'zone',
        'lat',
        'lon',
        'year',
        'month',
        'day',
        'date',
        'temp',
        'dwpt',
        'rhum',
        'prcp',
        'wdir',
        'wspd',
        'pres',
        'coco',
        'shape_length',
        'hour',
        'lon_factor',
        'lat_factor'
    ]
    for i,item in enumerate(keys):
        d[item] = custom_input_list[i]

    json_object = json.dumps(d, indent = 4) 
    return json_object

# Input Model
class Input(BaseModel):
    light: str
    weather: str 
    highway: str 
    type_of_collision: str
    zone: str
    lat: float 
    lon: float
    year: float
    month: float 
    day: float 
    date: float
    temp: float
    dwpt: float 
    rhum: float 
    prcp: float 
    wdir: float 
    wspd: float
    pres: float 
    coco: float 
    shape_length: float 
    hour: float
    lon_factor: float
    lat_factor: float

# Default Input List (for testing purposes)
default_input_list = ['day','clear-day','trunk','Unknown','Zone 2',
       19.017775,72.848129,2020.000000,12.000000,3.000000,31.000000,27.000000,18.000000,58.000000,
       0.000000,350.000000,5.400000,1012.000000,5.000000,314.801455,17.000000,0.848129,0.017775]

# API
app = FastAPI()
inp = pd.DataFrame(
    columns=[
        'light',
        'weather',
        'highway',
        'Type of Collision',
        'zone',
        'lat',
        'lon',
        'year',
        'month',
        'day',
        'date',
        'temp',
        'dwpt',
        'rhum',
        'prcp',
        'wdir',
        'wspd',
        'pres',
        'coco',
        'shape_length',
        'hour',
        'lon_factor',
        'lat_factor'
    ]
)
input_df = pd.DataFrame(
    columns=[
    'light_dawn',
    'light_day',
    'light_dusk',
    'light_night',
    'weather_None',
    'weather_clear-day',
    'weather_clear-night',
    'weather_cloudy',
    'weather_fog',
    'weather_partly-cloudy-day',
    'weather_partly-cloudy-night',
    'weather_wind',
    'highway_Unknown',
    'highway_bridleway',
    'highway_construction',
    'highway_footway',
    'highway_living_street',
    'highway_motorway',
    'highway_motorway_link',
    'highway_path',
    'highway_pedestrian',
    'highway_primary',
    'highway_primary_link',
    'highway_proposed',
    'highway_residential',
    'highway_secondary',
    'highway_secondary_link',
    'highway_service',
    'highway_tertiary',
    'highway_tertiary_link',
    'highway_trunk',
    'highway_unclassified',
    'Type of Collision_Bicycle to Bicycle',
    'Type of Collision_Boarding into bus',
    'Type of Collision_Break',
    'Type of Collision_Careless Driving',
    'Type of Collision_Cement Mixture spread on road',
    'Type of Collision_Driving zig zag',
    'Type of Collision_Electric Pole fell down',
    'Type of Collision_Fell Down from Bus',
    'Type of Collision_Fell down from running vehicle',
    'Type of Collision_Fixed Object',
    'Type of Collision_Head-on Collision',
    'Type of Collision_Head-on collision',
    'Type of Collision_Hit To Hand Cart',
    'Type of Collision_Hit from back',
    'Type of Collision_Hit from side',
    'Type of Collision_Hit to Asbestos',
    'Type of Collision_Hit to Auto Rickshaw and Pedestrain',
    'Type of Collision_Hit to Barricades',
    'Type of Collision_Hit to Bicycle and Pedestrain',
    'Type of Collision_Hit to Divider & Trailer',
    'Type of Collision_Hit to Divider and Pedestrian',
    'Type of Collision_Hit to Hand Cart',
    'Type of Collision_Hit to Iron Angle',
    'Type of Collision_Hit to Iron Barricades',
    'Type of Collision_Hit to Motor Cycle & Pedistrian',
    'Type of Collision_Hit to Parked Vehicle & Barricades',
    'Type of Collision_Hit to Parked Vehicle & Pedistrian',
    'Type of Collision_Hit to Parked Vehicles and Head on Collision',
    'Type of Collision_Hit to Pedestrian and Vehicle ',
    'Type of Collision_Hit to Pedistrian & Handcart',
    'Type of Collision_Hit to Vehicle & Divider',
    'Type of Collision_Hit to Vehicle and Pedistrian',
    'Type of Collision_Hit to Vehicle and Wall',
    'Type of Collision_Hit to barricades',
    'Type of Collision_Man Sleeping on Road',
    'Type of Collision_Motor Cycle Passenger Hits Vehicle',
    'Type of Collision_Motor Cycle hit to motor car opened door/Another Motor Car hit to Motor Cycle from behind',
    'Type of Collision_Parked Vehicle',
    'Type of Collision_Passenger Getting down from Bus',
    'Type of Collision_Passenger coming out of Taxi',
    'Type of Collision_Racing',
    'Type of Collision_Rash Driving',
    'Type of Collision_Rash Driving & Hit to Pedestrian',
    'Type of Collision_Rash Driving & Hit to Pedistrian',
    'Type of Collision_Rash Driving, zig zag',
    'Type of Collision_Rash driving',
    'Type of Collision_Run-off road',
    'Type of Collision_Slip',
    'Type of Collision_Sudden Break',
    'Type of Collision_Typre removed',
    'Type of Collision_Unknown',
    'Type of Collision_Vehicle Overturn',
    'Type of Collision_Vehicle to Bicycle',
    'Type of Collision_Vehicle to Pedestiran',
    'Type of Collision_Vehicle to Pedestrian',
    'Type of Collision_Vehicle to Vehicle',
    'Type of Collision_Woman Sleeping on Road',
    'Type of Collision_Zig zag driving',
    'zone_Unknown',
    'zone_Zone 1',
    'zone_Zone 2',
    'zone_Zone 3',
    'zone_Zone 4',
    'zone_Zone 5',
    'zone_Zone 6',
    'lat',
    'lon',
    'year',
    'month',
    'day',
    'date',
    'temp',
    'dwpt',
    'rhum',
    'prcp',
    'wdir',
    'wspd',
    'pres',
    'coco',
    'shape_length',
    'hour',
    'lon_factor',
    'lat_factor']
)

@app.post("/predict/")
async def predict(input: Input):
    # Preprocessing of Input
    input_row = list(input.dict().values())
    df2 = pd.DataFrame([[0.0]*input_df.shape[1]],columns=input_df.columns)
    df3 = pd.concat([input_df,df2])
    df3.fillna(0.0)
    count = 0
    for ele in input_row[:5]:
        df3[f"{inp.columns[count]}_{ele}"] = 1.0
        count+=1
    count = 5
    for num in input_row[5:]:
        df3[inp.columns[count]] = num
        count +=1

    scaler = pickle.load(open('scaler.sav', 'rb'))
    scaled_df3 = scaler.transform(df3)
    final_input = pd.DataFrame(scaled_df3, columns=df3.columns)

    # Depickling of model and Prediction
    loaded_model = pickle.load(open('XGB_Best_Model.pkl', 'rb'))
    xgc_proba = loaded_model.predict_proba(final_input)
    # print(xgc_proba)
    accident_proba = []
    for p in xgc_proba:
        cl = np.argmax(p)
        if cl == 0:
            accident_proba.append(np.average([p[0],p[1]], weights=[0.7, 0.3]))
        elif cl == 1 :
            accident_proba.append(np.average([p[0],p[1]], weights=[0.3, 0.7]))
        else:
            accident_proba.append(1 - p[2])
    # print(accident_proba)
    outcome = {0: 'Fatal', 1:'Injurious', 2:'Safe'}
    # Return prediction etc. whatever UI wants
    return {'accident_chance': round(accident_proba[0]*100,2), 'outcome': outcome[int(np.argmax(xgc_proba))]} 

# print(convert_input_list_to_json_input(default_input_list))