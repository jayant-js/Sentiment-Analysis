from fastapi import FastAPI
from Pipeline.pipeline_module import create_pipeline
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Annotated

@asynccontextmanager
async def lifespan(app: FastAPI):
    sentiment_pipeline = create_pipeline()
    sentiment_pipeline.fit('dummy')
    app.state.sentiment_pipeline = sentiment_pipeline
    yield

app = FastAPI()

class TextInput(BaseModel):
    text: Annotated[str, Field(..., min_length=2, description='Give the movie review...')]

@app.post('/get-sentiment')
def get_text(text: TextInput):  
    sentiment_pipeline = app.state.sentiment_pipeline
    encoded = sentiment_pipeline.named_steps['preprocessing'].transform(TextInput.text)
    prediction = sentiment_pipeline.named_steps['sentiment-prediction'].predict(encoded)
    return {'sentiment': int(prediction[0])}