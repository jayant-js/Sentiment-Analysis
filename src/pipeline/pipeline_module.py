from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scipy.special import softmax
import torch
from config.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "roberta-sentiment-tokenizer"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "roberta-sentiment-analysis"

logger.info(f"Paths configured. Model will be loaded from: {MODEL_PATH}")
logger.info(f"Tokenizer will be loaded from: {TOKENIZER_PATH}")

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer_path = TOKENIZER_PATH
        self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_path)
    
    def fit(self, X, y=None):
        return self 

    def transform(self, X: str, y = None):
        if not isinstance(X, str):
            raise TypeError('Input X must be a string.')
        x_processed = X.strip().lower() 
        encoded = self.tokenizer([x_processed], padding=True, truncation=True, max_length = 128, return_tensors = 'pt')
        encoded_data = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
        }
        return encoded_data
    
class SentimentPredictor(BaseEstimator):
    def __init__(self):
        self.model_path = MODEL_PATH
        self.model: RobertaForSequenceClassification
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_path, from_tf=True)
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X, y=None):
        with torch.no_grad():
            output = self.model(**X)
        logits = output.logits
        probs = softmax(logits.numpy(), axis=1)
        preds = np.argmax(probs, axis = 1)
        return probs, preds
    
def create_pipeline():
    return Pipeline([
        ('preprocessing', PreprocessingTransformer()), 
        ('sentiment-prediction', SentimentPredictor())
    ])