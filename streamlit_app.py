import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from text_preprocessing import clean_text

# ==========================
# Config
# ==========================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================

model = load_model("sentiment_lstm.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50

# ==========================
# Header
# ==========================

st.title("💬 Sentiment Analysis App")

st.markdown(
    """
    ### Welcome Asyifa 👋

    Analyze text sentiment using an LSTM Deep Learning model.

    Enter a review, comment, or opinion below and the model will predict whether it is positive or negative.
    """
)

st.divider()

# ==========================
# Input
# ==========================

text_input = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Example: Pelayanannya sangat baik dan makanannya enak..."
)

# ==========================
# Predict
# ==========================

if st.button("Analyze Sentiment"):

    if text_input.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    clean = clean_text(text_input)

    sequence = tokenizer.texts_to_sequences([clean])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(padded, verbose=0)[0][0]

    confidence = prediction * 100

    if prediction >= 0.5:

        sentiment = "Positive 😊"
        color = "green"

    else:

        sentiment = "Negative 😞"
        confidence = (1 - prediction) * 100
        color = "red"

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
    "Sentiment",
    sentiment,
    f"{confidence:.2f}% confidence" 
    )

    st.progress(float(confidence) / 100)

    st.write(
        f"**Confidence Score:** {confidence:.2f}%"
    )

    if color == "green":
        st.success(
            "The model predicts that this text has a positive sentiment."
        )
    else:
        st.error(
            "The model predicts that this text has a negative sentiment."
        )