import os, sys

import certifi

from dotenv import load_dotenv


import pymongo
import pandas as pd

from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging
from flushot.pipeline.training_pipeline import TrainingPipeline
from flushot.utils.main_utils.utils import load_object
from flushot.utils.ml_utils.model.estimator import NetworkModel

from flushot.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from uvicorn import run as app_run
from starlette.responses import RedirectResponse

load_dotenv()
mongo_db_url = os.getenv('MONGO_URI')

ca = certifi.where()

client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = client[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

templates = Jinja2Templates(directory='./templates')

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')

@app.get('/train')
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response('Training is successful')

    except Exception as e:
        raise FluShotException(e, sys)

@app.get('/predict')
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object('final_model/preprocessor.pkl')
        final_model = load_object('final_model/model.pkl')
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])

        y_pred = network_model.predict(df)
        print(y_pred)

        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse('table.html', {'request': request, 'table': table_html})
    except Exception as e:
        raise FluShotException(e, sys)

if __name__ == '__main__':
    app_run(app, host='0.0.0.0', port= 8000)