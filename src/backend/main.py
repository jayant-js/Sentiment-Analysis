from fastapi import FastAPI, Request, HTTPException
from pipeline.pipeline_module import create_pipeline
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import numpy as np
from config.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Application startup: Loading models and pipeline')
    sentiment_pipeline = create_pipeline()
    sentiment_pipeline.fit('dummy')
    app.state.sentiment_pipeline = sentiment_pipeline
    logger.info("Models and pipeline loaded successfully")
    yield

app = FastAPI(lifespan=lifespan)

class TextInput(BaseModel):
    review: str = Field(..., min_length=2, description='Give the movie review...')

@app.get('/')
def intro():
    return {'message':'This application predicts the sentiment of the movie review'}

@app.post('/get-sentiment')
def get_text(text: TextInput, request:Request):  
    review = text.review
    if not review.strip():
        raise HTTPException(status_code=400, detail='Review cannot be empty')
    logger.info(f'Recieved review: {review}')
    try:
        sentiment_pipeline = request.app.state.sentiment_pipeline
        logger.info(f"Preprocessing the incoming request: {review}")
        encoded = sentiment_pipeline.named_steps['preprocessing'].transform(review)
        logger.info('Text preprocessed, now predicting the sentiment...')
        probs, preds = sentiment_pipeline.named_steps['sentiment-prediction'].predict(encoded)
        logger.info('Sentiment extracted')
    except Exception as e:
        logger.error(f'Error finding the sentiment: {e}')
        raise HTTPException(status_code=500, detail=str(e))

    pred_class = int(preds[0])
    confidence_score = float(np.max(probs[0]))
    all_confidence = probs[0].tolist()

    return {
        'sentiment':pred_class,
        'confidence':confidence_score,
        'all_confidence':all_confidence
    }