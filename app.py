import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="BBC News Classification",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
    <style>
        html, body, [class*="css"], .stApp, .main, .block-container,
        section[data-testid="stSidebar"], header, footer,
        div[data-testid="stToolbar"], div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .stTextArea textarea {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        .stSelectbox div[data-baseweb="select"],
        div[data-baseweb="popover"], ul[data-baseweb="menu"] {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        .stButton > button {
            background-color: #222222 !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        .stButton > button:hover { background-color: #444444 !important; }
        .stSuccess, .stInfo, .stWarning {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        p, h1, h2, h3, h4, label, span { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Load Saved Files
# -----------------------------------------------------

@st.cache_resource
def load_files():

    tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

    tokenizer = pickle.load(open("models/tokenizer.pkl", "rb"))

    label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

    tfidf_model = tf.keras.models.load_model("models/tfidf_model.keras")

    rnn_model = tf.keras.models.load_model("models/rnn_model.keras")

    lstm_model = tf.keras.models.load_model("models/lstm_model.keras")

    gru_model = tf.keras.models.load_model("models/gru_model.keras")

    return (
        tfidf,
        tokenizer,
        label_encoder,
        tfidf_model,
        rnn_model,
        lstm_model,
        gru_model,
    )


(
    tfidf,
    tokenizer,
    label_encoder,
    tfidf_model,
    rnn_model,
    lstm_model,
    gru_model,
) = load_files()

MAX_LENGTH = 300

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("📰 BBC News Text Classification")

st.write(
    "Predict the category of a BBC news article using "
    "**TF-IDF**, **RNN**, **LSTM**, or **GRU**."
)

# -----------------------------------------------------
# Model Selection
# -----------------------------------------------------

model_name = st.selectbox(
    "Choose Model",
    [
        "TF-IDF",
        "RNN",
        "LSTM",
        "GRU"
    ]
)

# -----------------------------------------------------
# Text Input
# -----------------------------------------------------

user_text = st.text_area(
    "Enter News Article",
    height=220
)

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

if st.button("Predict Category"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    # ---------------- TF-IDF ----------------

    if model_name == "TF-IDF":

        vector = tfidf.transform([user_text]).toarray()

        prediction = tfidf_model.predict(vector)

    # ---------------- Deep Learning ----------------

    else:

        seq = tokenizer.texts_to_sequences([user_text])

        padded = pad_sequences(
            seq,
            maxlen=MAX_LENGTH,
            padding="post"
        )

        if model_name == "RNN":
            prediction = rnn_model.predict(padded)

        elif model_name == "LSTM":
            prediction = lstm_model.predict(padded)

        else:
            prediction = gru_model.predict(padded)

    predicted_class = np.argmax(prediction)

    label = label_encoder.inverse_transform([predicted_class])[0]

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Category : **{label.upper()}**")

    st.info(f"Confidence : **{confidence:.2f}%**")

    st.subheader("Prediction Probabilities")

    for cls, prob in zip(
        label_encoder.classes_,
        prediction[0]
    ):
        st.write(f"**{cls}** : {prob*100:.2f}%")