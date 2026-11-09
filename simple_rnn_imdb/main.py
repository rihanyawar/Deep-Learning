
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN


# ==========================================
# 2. CONFIGURATION
# ==========================================

MAX_FEATURES = 10000
MAX_LENGTH = 500

MODEL_PATH = "simple_rnn_imdb.h5"


# ==========================================
# 3. LOAD IMDB WORD INDEX
# ==========================================

word_index = imdb.get_word_index()

# Reverse word index (for decoding)
reverse_word_index = {
    value + 3: key for key, value in word_index.items()
}


# ==========================================
# 4. HANDLE KERAS COMPATIBILITY
# ==========================================

class CompatibleSimpleRNN(SimpleRNN):

    def __init__(self, *args, **kwargs):

        kwargs.pop("time_major", None)

        super().__init__(*args, **kwargs)


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_sentiment_model():

    model = load_model(
        MODEL_PATH,
        custom_objects={
            "SimpleRNN": CompatibleSimpleRNN
        },
        compile=False
    )

    return model


try:

    model = load_sentiment_model()
    st.success("Model loaded successfully!")

except Exception as e:

    st.error(f"Error loading model: {e}")
    st.stop()


# ==========================================
# 6. PREPROCESS USER INPUT
# ==========================================

def preprocess_text(text):

    words = text.lower().split()

    encoded_review = []

    for word in words:

        index = word_index.get(word)

        if index is None or index >= MAX_FEATURES:
            encoded_review.append(2)

        else:
            encoded_review.append(index + 3)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH,
        padding="pre",
        truncating="pre"
    )

    return padded_review


# ==========================================
# 7. DECODE REVIEW (OPTIONAL)
# ==========================================

def decode_review(encoded_review):

    return " ".join(
        reverse_word_index.get(index, "?")
        for index in encoded_review
        if index > 3
    )


# ==========================================
# 8. STREAMLIT INTERFACE
# ==========================================

st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below to classify "
    "its sentiment as Positive or Negative."
)

st.divider()

user_input = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was absolutely amazing...",
    height=150
)


# ==========================================
# 9. MAKE PREDICTION
# ==========================================

if st.button("Classify Sentiment", type="primary"):

    if not user_input.strip():

        st.warning("Please enter a movie review first.")

    else:

        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(
            preprocessed_input,
            verbose=0
        )

        score = float(np.asarray(prediction).reshape(-1)[0])

        if score > 0.5:

            sentiment = "Positive"
            st.success("Positive Sentiment 😊")

        else:

            sentiment = "Negative"
            st.error("Negative Sentiment 😞")

        st.metric(
            label="Prediction Score",
            value=f"{score:.4f}"
        )

        st.write(f"**Predicted Sentiment:** {sentiment}")

        st.progress(score)
