import streamlit as st
import requests

text = st.input_text('Give your review here')
if text:
    if st.button():
        response = requests.post(f'http://localhost:8000/input-text/{text}')
        if response.status_code == 200:
            sentiment = requests.get('http://localhost:8000/get-sentiment')
            if sentiment == 1:
                

