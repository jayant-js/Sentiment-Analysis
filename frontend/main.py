import streamlit as st
import requests

text = st.text_input('Give your review here')
if st.button('Predict'):
    if text:
        with st.spinner('Getting Sentiment...'):
            response = requests.post('http://localhost:8000/get-sentiment', json={'text': text})
            if response.status_code == 200:
                sentiment = response.json().get('sentiment')
                if sentiment == 0:
                    st.write('Negative')
                elif sentiment == 1:
                    st.write('Positive')
            else:
                st.error(response.status_code)
    else:
        st.warning('Give the input')