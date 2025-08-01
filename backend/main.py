from fastapi import FastAPI, Path
from Pipeline.pipeline_module import create_pipeline
from pydantic import BaseModel, Field
from typing import Annotated

app = FastAPI()

class TextInput(BaseModel):
    text: Annotated[str, Field(..., min_length=2)]

@app.on_event('startup')
def load_pipeline():
    global sentiment_pipeline 
    sentiment_pipeline = create_pipeline()
    sentiment_pipeline.fit('dummy input')

@app.post('/input-text')
def get_text(text: TextInput):  
    global sentiment_pipeline
    sentiment_pipeline.named_steps['preprocessing'].transform(text.text)
    return {'message':'Text preprocessed sucessfully'}

@app.get('/get-sentiment')
def get_sentiment(): 
    global sentiment_pipeline
    encoded = sentiment_pipeline.named_steps['preprocessing'].encoded_data   
    prediction = sentiment_pipeline.named_steps['sentiment-prediction'].predict(encoded)
    return {'sentiment': int(prediction[0])}