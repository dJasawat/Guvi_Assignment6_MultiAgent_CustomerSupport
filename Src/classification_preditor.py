import os
import pickle
import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# get environment variables for model and data paths
MODEL_PATH = os.getenv("MODEL_PATH")
DATA_PATH = os.getenv("DATA_PATH")

stop_words = set(stopwords.words('english'))
class ClassifierPredictor:
    def __init__(self):
    # =========================================================
    # 3. SENTIMENT PREDICTION
    # =========================================================

    # Load Sentiment TF-IDF Vectorizer
        self.sentiment_vectorizer_path = os.path.join(
          MODEL_PATH,
         "tfidf_vectorizer_sentiment.pkl"
        )

        with open(self.sentiment_vectorizer_path, "rb") as f:
          self.sentiment_vectorizer = pickle.load(f)

        # Load Best Sentiment Model
        self.sentiment_model_path = os.path.join(
                MODEL_PATH,
                "Sentiment_BestModel_Logistic Regression.pkl"
                )
        
        with open(self.sentiment_model_path, "rb") as f:
            self.sentiment_model = pickle.load(f)

    def predict(self, user_query):
        # =========================================================
        # 1. PREPROCESS INPUT
        # =========================================================
        cleaned_text = self.clean_data(user_query)
        
         # Transform query
        sentiment_vectorized_text = (
            self.sentiment_vectorizer.transform([cleaned_text])
            )
        # Implementation for prediction
         # Predict Sentiment
        sentiment_prediction = self.sentiment_model.predict(
                sentiment_vectorized_text
            )[0]

        sentiment_probabilities = self.sentiment_model.predict_proba(
                sentiment_vectorized_text)[0]
        confifdence= float(max(sentiment_probabilities))
        return sentiment_prediction, confifdence
        
        
    def clean_data(self, text):
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

