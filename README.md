# News-Classification

# 📰 BBC News Text Classification using Machine Learning & Deep Learning

## 📌 Project Overview

This project classifies BBC news articles into one of five categories using both Machine Learning and Deep Learning techniques. The application provides a Streamlit-based web interface where users can enter a news article and choose a trained model to predict its category.

The project compares the performance of the following models:

- Logistic Regression
- TF-IDF Neural Network
- Simple RNN
- LSTM
- GRU

The results show that **Logistic Regression with TF-IDF** achieves the highest accuracy on the BBC News dataset.

# 📂 Dataset

**Dataset:** BBC News Classification Dataset

The dataset contains news articles belonging to five categories:

- Business
- Entertainment
- Politics
- Sport
- Tech
  

# 🚀 Features

- Text Classification using Machine Learning and Deep Learning
- TF-IDF Feature Extraction
- Word Tokenization
- Text Padding
- Logistic Regression Classifier
- TF-IDF Neural Network
- Simple RNN
- LSTM Network
- GRU Network
- Interactive Streamlit Web Application
- Model Comparison


# 🛠 Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Pickle


# 📁 Project Structure

```
BBC-News-Classification/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── bbc-text(2).csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── tfidf_nn.keras
│   ├── rnn.keras
│   ├── lstm.keras
│   └── gru.keras
│
└── screenshots/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/BBC-News-Classification.git
```

Move into the project directory

```bash
cd BBC-News-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Models

Run

```bash
python train_model.py
```

This script will:

- Load the dataset
- Perform preprocessing
- Generate TF-IDF features
- Train Logistic Regression
- Train TF-IDF Neural Network
- Train RNN
- Train LSTM
- Train GRU
- Save all trained models inside the **models/** folder


# ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 🧠 Models Used

## 1. Logistic Regression

- TF-IDF Feature Extraction
- Multi-class Classification
- Fast Training
- High Accuracy

---

## 2. TF-IDF Neural Network

Architecture

```
Input Layer
↓

Dense (256)

↓

Dense (128)

↓

Output Layer (Softmax)
```

---

## 3. Simple RNN

Architecture

```
Embedding

↓

SimpleRNN

↓

Dense

↓

Output
```

---

## 4. LSTM

Architecture

```
Embedding

↓

LSTM

↓

Dense

↓

Output
```

---

## 5. GRU

Architecture

```
Embedding

↓

GRU

↓

Dense

↓

Output
```

---

# 📊 Model Performance

| Model | Accuracy |
|---------|----------|
| Logistic Regression | **98%** |
| TF-IDF Neural Network | **98%** |
| LSTM | 68% |
| GRU | 68% |
| Simple RNN | 29% |

**Best Model:** Logistic Regression with TF-IDF

---

# 📈 Workflow

```
Dataset

↓

Text Cleaning

↓

Train-Test Split

↓

Feature Extraction

├── TF-IDF

└── Tokenization

↓

Model Training

├── Logistic Regression

├── TF-IDF Neural Network

├── RNN

├── LSTM

└── GRU

↓

Model Evaluation

↓

Streamlit Deployment
```

---

# 🖥 Streamlit Application

The application allows users to:

- Enter BBC news text
- Select a prediction model
- Predict the news category
- View prediction confidence
- Compare different models

---

# 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

---

# 📌 Future Enhancements

- Bidirectional LSTM
- BERT Text Classification
- RoBERTa
- DistilBERT
- Attention Mechanism
- Hyperparameter Optimization
- Model Explainability using SHAP/LIME

---

# 📚 Libraries Used

- streamlit
- tensorflow
- scikit-learn
- pandas
- numpy
- pickle

  
Machine Learning | Deep Learning | NLP | Python | Streamlit
