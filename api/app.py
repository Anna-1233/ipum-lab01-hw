import joblib
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
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

    @field_validator("text")
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("text cannot be empty or whitespace")

        return v


class SentimentResponse(BaseModel):
    prediction: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API. Go to /docs for Swagger UI."}


@app.post("/predict")
def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    embedding_vector = transformer.encode([request.text])
    prediction = classifier.predict(embedding_vector)[0]
    sentiment = SENTIMENT_CLASS.get(int(prediction))
    return SentimentResponse(prediction=sentiment)
