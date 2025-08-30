import streamlit as st
import requests
import numpy as np
import os
from config.logger import get_logger

logger = get_logger(__name__)   
api_url = os.getenv('API_URL', "http://localhost:8000/get-sentiment")
text = st.text_input('Give your review here')
if st.button('Predict'):
    if text:
        with st.spinner('Getting Sentiment...'):
            response = requests.post('http://localhost:8000/get-sentiment', json={'text': text})
            if response.status_code == 200:
                response_json = response.json()
                sentiment = response_json.get('sentiment')
                confidence = round(response_json.get('confidence'), 2)
                all_confidence = response_json.get('all_confidence')
                if sentiment == 0:
                    st.write('Negative: ', confidence)
                elif sentiment == 1:
                    st.write('Positive: ', confidence)
                all_class_probs = {'Negative': round(all_confidence[0], 2), 'Positive': round(all_confidence[1], 2)}
                st.write('Confidence scores for all classes: ', all_class_probs)
            else:
                st.error(response.status_code)
    else:
        st.warning('Give the input')