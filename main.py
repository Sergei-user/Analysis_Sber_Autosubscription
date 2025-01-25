import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

df = pd.read_csv('data/df_session_target.csv')

with open('model/sber_cars_sub_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: float
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    client_id: float
    session_id: str
    target_action: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def status():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    data = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(data)

    return {
        'client_id': form.client_id,
        'session_id': form.session_id,
        'target_action': y[0]
    }
