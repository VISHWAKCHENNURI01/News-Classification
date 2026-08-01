import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

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

@st.cache_resource
def train_models():
    df = pd.read_csv("bbc-text.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["category"])
    X = df["text"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)

    lr   = LogisticRegression(max_iter=1000)
    lr.fit(X_train_tfidf, y_train)

    tfidf_nn = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=50, random_state=42)
    tfidf_nn.fit(X_train_tfidf, y_train)

    rnn  = SGDClassifier(max_iter=100, random_state=42)
    rnn.fit(X_train_tfidf, y_train)

    lstm = PassiveAggressiveClassifier(max_iter=100, random_state=42)
    lstm.fit(X_train_tfidf, y_train)

    gru  = Perceptron(max_iter=100, random_state=42)
    gru.fit(X_train_tfidf, y_train)

    return tfidf, encoder, lr, tfidf_nn, rnn, lstm, gru

with st.spinner("Loading models..."):
    tfidf, encoder, lr_model, tfidf_nn_model, rnn_model, lstm_model, gru_model = train_models()

MODEL_MAP = {
    "Logistic Regression": lr_model,
    "TF-IDF Neural Network": tfidf_nn_model,
    "RNN": rnn_model,
    "LSTM": lstm_model,
    "GRU": gru_model,
}

st.title("📰 BBC News Text Classification")
st.write("Predict the category of a BBC news article using **Logistic Regression**, **TF-IDF Neural Network**, **RNN**, **LSTM**, or **GRU**.")

model_name = st.selectbox("Choose Model", list(MODEL_MAP.keys()))
user_text  = st.text_area("Enter News Article", height=220)

if st.button("Predict Category"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    vector = tfidf.transform([user_text])
    model  = MODEL_MAP[model_name]

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vector)[0]
        pred_class = np.argmax(proba)
        confidence = proba[pred_class] * 100
        label      = encoder.inverse_transform([pred_class])[0]
        st.success(f"Predicted Category : **{label.upper()}**")
        st.info(f"Confidence : **{confidence:.2f}%**")
        st.subheader("Prediction Probabilities")
        for cls, prob in zip(encoder.classes_, proba):
            st.write(f"**{cls}** : {prob*100:.2f}%")
    else:
        pred_class = model.predict(vector)[0]
        label      = encoder.inverse_transform([pred_class])[0]
        st.success(f"Predicted Category : **{label.upper()}**")
