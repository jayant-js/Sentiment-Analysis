import streamlit as st
import requests
from fastapi import HTTPException

text = st.text_input('Give your review here')
if st.button('Predict'):
    if text:
        with st.spinner('Preprocessing the Input...'):
            response = requests.post(f'http://localhost:8000/input-text', json={'text': text})
        if response.status_code == 200: 
            with st.spinner('Getting sentiment...'):
                response = requests.get('http://localhost:8000/get-sentiment')
                if response.status_code == 200:
                    sentiment = response.json().get('sentiment')
                    if sentiment == 1:
                        st.write('Positive')
                    elif sentiment == 0:
                        st.write('Negative')
                else:
                    st.error('Error while getting sentiment.')
        else:
            st.error('Error during preprocessing')
    else:
        st.warning('Please enter some text')