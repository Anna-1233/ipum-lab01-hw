import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


SENTIMENT_CLASS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


TRANSFORMER_PATH = "model/sentence_transformer.model"
CLASSIFIER_PATH = "model/classifier.joblib"


transformer = SentenceTransformer(TRANSFORMER_PATH)
classifier = joblib.load(CLASSIFIER_PATH)


app = FastAPI(title="Sentiment Analysis API")


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    prediction: str


@app.post("/predict")
def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    embedding_vector = transformer.encode([request.text])
    prediction = classifier.predict(embedding_vector)[0]
    sentiment = SENTIMENT_CLASS.get(int(prediction))
    return SentimentResponse(prediction=sentiment)
