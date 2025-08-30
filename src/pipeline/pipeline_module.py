from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from scipy.special import softmax
from config.logger import get_logger

logger = get_logger(__name__)

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer_path = r'D:/Projects/Sentiment Analysis using BERT/backend/Pipeline/roberta-sentiment-tokenizer'):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_path)
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y = None):
        X = list(X.strip())
        X_lower = [text.lower() for text in X]  
        encoded = self.tokenizer(X_lower, padding=True, truncation=True, max_length = 128, return_tensors = 'tf')
        encoded_data = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
        }
        return encoded_data
    
class SentimentPredictor(BaseEstimator):
    def __init__(self, model_path = r'D:/Projects/Sentiment Analysis using BERT/backend/Pipeline/roberta-sentiment-analysis'):
        self.model_path = model_path
        self.model = TFRobertaForSequenceClassification.from_pretrained(self.model_path)
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X, y=None):
        output = self.model.predict(X)
        logits = output.logits
        probs = softmax(logits, axis=1)
        preds = np.argmax(probs, axis = 1)
        return probs, preds
    
def create_pipeline():
    return Pipeline([
        ('preprocessing', PreprocessingTransformer()), 
        ('sentiment-prediction', SentimentPredictor())
    ])