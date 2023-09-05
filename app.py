import streamlit as st
import altair as alt
from sentiment_analyzer import SentimentAnalyzer

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Define the path to the downloaded repository containing the model
REPO_PATH = "twitter-roberta-base-sentiment-latest"

# Create an instance of the SentimentAnalyzer class
analyzer = SentimentAnalyzer(REPO_PATH)

# Get the text parameter from the URL
params = st.experimental_get_query_params()
text = params.get("text")


if text:
    
    # Get the actual value of the text parameter
    text = text[0]
    
    # Analyze the text from the URL parameter
    label, score = analyzer.analyze_text(text)
    st.write({"model_output": label, "confidence_score": score})

