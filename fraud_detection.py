import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Function to clean text
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())


# Function to train the fraud detection model using Naive Bayes
def train_model():
    # Load the updated dataset
    data = pd.read_csv('fraud_data.csv')
    data['Cleaned_Message'] = data['Message'].apply(clean_text)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Cleaned_Message'])
    y = data['Label']

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer


# Function to predict the fraud type for a new message
def predict_fraud(model, vectorizer, message):
    cleaned_message = clean_text(message)
    message_vec = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_vec)
    return prediction[0]
