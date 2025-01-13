from flask import Flask, request, render_template
from fraud_detection import train_model, predict_fraud

app = Flask(__name__)

# Load the Naive Bayes model and vectorizer
model, vectorizer = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentence():
    sentence = request.form['sentence']  # User input sentence
    prediction = predict_fraud(model, vectorizer, sentence)
    return render_template('index.html', result=f"The sentence is classified as: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
