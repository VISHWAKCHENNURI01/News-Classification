import streamlit as st
import pickle
import numpy as np

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
def load_files():
    tfidf      = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    encoder    = pickle.load(open("models/label_encoder.pkl", "rb"))
    lr_model   = pickle.load(open("models/logistic_regression.pkl", "rb"))
    svm_model  = pickle.load(open("models/svm.pkl", "rb"))
    mlp_model  = pickle.load(open("models/mlp.pkl", "rb"))
    nb_model   = pickle.load(open("models/naive_bayes.pkl", "rb"))
    return tfidf, encoder, lr_model, svm_model, mlp_model, nb_model

tfidf, encoder, lr_model, svm_model, mlp_model, nb_model = load_files()

MODEL_MAP = {
    "Logistic Regression": lr_model,
    "SVM": svm_model,
    "MLP Neural Network": mlp_model,
    "Naive Bayes": nb_model,
}

st.title("📰 BBC News Text Classification")
st.write("Predict the category of a BBC news article using ML models.")

model_name = st.selectbox("Choose Model", list(MODEL_MAP.keys()))

user_text = st.text_area("Enter News Article", height=220)

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
