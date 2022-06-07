######################
# Import libraries
######################

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
import re # regular expressions
import joblib

######################
# Page Title
######################

image = Image.open('apps/movie_review_app/imdb_logo.png')

st.image(image, use_column_width=True)

st.write("""
# Movie Review Sentiment Web App
This app classifies a movie review as positive or negative
***
""")

############################
#  Load classifier
############################

def load_model(model_path):
    model = joblib.load(model_path)
    #st.write('model succesfully loaded!')
    return model

model_path = 'apps/movie_review_app/log_reg_model.pkl'

if 'model' not in st.session_state:
    st.session_state['model'] = load_model(model_path)


######################
# Input Text Box
######################

#st.sidebar.header('Enter DNA sequence')
st.header('Enter the review')

sequence_input = "This is the worst work ever of Daniel Day Lewis..... I can not believe that in the same year he made this awful movie and My left foot..... Please stay away from this movie....this is a movie only for Argentine people as a curiosity... The plot is impossible to understand...... The writer thinks that in Argentine all the people speaks in english... Of course the Patagonia bring a very good frame for the photo shooting of the film, but that is not enough reason to see this movie.... I repeat , only if you are very fan of Daniel Day Lewis, or if you want to see the south of Argentine, part of the Patagonia, and you do not have enough money to travel yourself......."

#sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
sequence = st.text_area("Sequence input", sequence_input, height=250)


def simple_preprocessor(text):
    text = re.sub('<.*?>','',text)      # se eliminan los tags html
    text = re.sub('[\W]+', ' ', text)  # se eliminan caracteres 'non-words' 
                                        # Words characters are  a letter or digit or underbar [a-zA-Z0-9_].)
    text = text.lower()                #
    return text


sequence = simple_preprocessor(sequence)
st.write("""
***
""")
prediction = st.session_state['model'].predict([sequence])


## Prints the input sequence
st.header('MODEL INPUT')
st.write(sequence)

st.header('MODEL OUTPUT (Sentiment)')
st.write('Positive review' if prediction else 'Negative review')


