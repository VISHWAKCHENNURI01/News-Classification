import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------
# Create Models Folder
# --------------------------------------------------------

os.makedirs("models", exist_ok=True)

# --------------------------------------------------------
# Load Dataset
# --------------------------------------------------------

df = pd.read_csv(r"D:\Streamlit\NLP Project\bbc-text.csv")

print(df.head())
print(df.shape)

# --------------------------------------------------------
# Encode Labels
# --------------------------------------------------------

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["category"])

pickle.dump(
    encoder,
    open("models/label_encoder.pkl", "wb")
)

X = df["text"]
y = df["label"]

# --------------------------------------------------------
# Train Test Split
# --------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# TF-IDF
# ==========================================================

print("\nCreating TF-IDF Features...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

pickle.dump(
    tfidf,
    open("models/tfidf_vectorizer.pkl", "wb")
)

# ==========================================================
# Logistic Regression
# ==========================================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(
    X_train_tfidf,
    y_train
)

pred = lr.predict(X_test_tfidf)

print("\nLogistic Regression Accuracy")

print(accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

pickle.dump(
    lr,
    open("models/logistic_regression.pkl", "wb")
)

# ==========================================================
# TF-IDF Neural Network
# ==========================================================

print("\nTraining TF-IDF Neural Network...")

tfidf_model = Sequential()

tfidf_model.add(Dense(
    256,
    activation="relu",
    input_shape=(5000,)
))

tfidf_model.add(Dense(
    128,
    activation="relu"
))

tfidf_model.add(Dense(
    len(encoder.classes_),
    activation="softmax"
))

tfidf_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

tfidf_model.fit(
    X_train_tfidf.toarray(),
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

pred = tfidf_model.predict(X_test_tfidf.toarray())

pred = np.argmax(pred, axis=1)

print("TF-IDF NN Accuracy")

print(accuracy_score(y_test, pred))

tfidf_model.save("models/tfidf_nn.keras")

# ==========================================================
# Tokenizer
# ==========================================================

MAX_WORDS = 10000
MAX_LENGTH = 300

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

pickle.dump(
    tokenizer,
    open("models/tokenizer.pkl", "wb")
)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding="post"
)

# ==========================================================
# Function
# ==========================================================

def train_dl_model(model, name):

    print(f"\nTraining {name}")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_pad,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    pred = model.predict(X_test_pad)

    pred = np.argmax(pred, axis=1)

    acc = accuracy_score(y_test, pred)

    print(f"{name} Accuracy :", acc)

    model.save(f"models/{name.lower()}.keras")

# ==========================================================
# RNN
# ==========================================================

rnn = Sequential()

rnn.add(Embedding(MAX_WORDS,128))

rnn.add(SimpleRNN(64))

rnn.add(Dense(64,activation="relu"))

rnn.add(Dense(len(encoder.classes_),activation="softmax"))

train_dl_model(rnn,"rnn")

# ==========================================================
# LSTM
# ==========================================================

lstm = Sequential()

lstm.add(Embedding(MAX_WORDS,128))

lstm.add(LSTM(64))

lstm.add(Dense(64,activation="relu"))

lstm.add(Dense(len(encoder.classes_),activation="softmax"))

train_dl_model(lstm,"lstm")

# ==========================================================
# GRU
# ==========================================================

gru = Sequential()

gru.add(Embedding(MAX_WORDS,128))

gru.add(GRU(64))

gru.add(Dense(64,activation="relu"))

gru.add(Dense(len(encoder.classes_),activation="softmax"))

train_dl_model(gru,"gru")

print("\nAll Models Trained Successfully.")