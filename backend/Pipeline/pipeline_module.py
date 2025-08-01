from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer_path = r'D:/Projects/Sentiment Analysis using BERT/backend/Pipeline/bert-sentiment-tokenizer'):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.encoded_data = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X: str, y = None):
        X = [X.strip()]
        X_lower = [text.lower() for text in X]  
        encoded = self.tokenizer(X_lower, padding=True, truncation=True, max_length = 128, return_tensors = 'tf')
        self.encoded_data = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
        }
        return self.encoded_data
    
class SentimentPredictor(BaseEstimator):
    def __init__(self, model_path = r'D:/Projects/Sentiment Analysis using BERT/backend/Pipeline/bert-sentiment-analysis'):
        self.model_path = model_path
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_path)
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X, y=None):
        output = self.model.predict(X)
        logits = output.logits
        return np.argmax(logits, axis = 1)
    
def create_pipeline():
    return Pipeline([
        ('preprocessing', PreprocessingTransformer()), 
        ('sentiment-prediction', SentimentPredictor())
    ])