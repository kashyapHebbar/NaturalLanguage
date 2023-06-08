from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
from typing import List
from scipy.sparse import csr_matrix
import random

app = FastAPI()

model = xgb.Booster()
model.load_model('xgboost_model.model')

ACTUAL_CLASS_NAMES_MOD = {
    0: "esteem",
    1: "amusement",
    2: "reassurance",
    3: "empathy",
    4: "elation",
    5: "frustration",
    6: "dissatisfaction",
    7: "apprehension",
    8: "sorrow",
    9: "confusion",
    10: "curiosity",
    11: "wonder",
    12: "desire",
    13: "neutral"
}

random.seed(123)


class FeatureVector(BaseModel):
    feature_vector: List[List[float]]


def get_prediction(data_vec):
    sparse_matrix = csr_matrix(data_vec)
    dtest = xgb.DMatrix(sparse_matrix)
    prediction = model.predict(dtest)
    pred_value = prediction[0]
    return ACTUAL_CLASS_NAMES_MOD[pred_value]


@app.post("/classifier/")
def classify_text(data: FeatureVector):
    return {'pred': get_prediction(data.feature_vector)}
