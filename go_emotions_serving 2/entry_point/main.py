from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import json
import requests

app = FastAPI()


class Text(BaseModel):
    text: str


@app.post("/predict")
async def predict(text: Text):
    # preprocess
    preprocessed_text_response = requests.post(
        'http://preprocessing_service:2000/preprocess', json=text.dict())
    preprocessed_text_data = preprocessed_text_response.json()

    # featurization
    featurization_response = requests.post('http://featurization_service:3000/featurization', json={
                                           "preprocessed_text": preprocessed_text_data["preprocessed_text"]})
    featurization_data = featurization_response.json()

    # NLP Algorithm
    classifier_response = requests.post('http://nlp_algorithms_service:4000/classifier', json={
                                        "feature_vector": featurization_data["featurized_vector"]})
    classifier_data = classifier_response.json()

    # log interaction
    interaction_log = {
        "input": text.text,
        "prediction": classifier_data['pred'],
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open('/app/logs/interaction_logs.txt', 'a') as f:
        f.write(json.dumps(interaction_log) + "\n")

    return {"prediction": classifier_data['pred']}


@app.get("/predict")
def get_predict():
    return {"Status": "OK"}
