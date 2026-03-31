from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Sentiment Analysis API")


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    prediction: str


@app.post("/predict")
def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    return SentimentResponse(prediction="positive")
