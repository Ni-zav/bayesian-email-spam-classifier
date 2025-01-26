from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

app = Flask(__name__)

try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()

    if not os.path.exists('nb_classifier.pkl') or not os.path.exists('words.pkl'):
        raise FileNotFoundError("Required model files not found")

    with open('nb_classifier.pkl', 'rb') as model_file:
        nb_classifier = pickle.load(model_file)

    with open('words.pkl', 'rb') as words_file:
        words = pickle.load(words_file)

    word_to_idx = {word.lower(): idx for idx, word in enumerate(words)}

except Exception as e:
    print(f"Initialization error: {str(e)}")
    raise

def preprocess_text(text):
    try:
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [porter.stem(word) for word in tokens]
        return tokens
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return []

def text_to_word_counts(tokens):
    try:
        word_counts = np.zeros(len(words))
        for word in tokens:
            if word in word_to_idx:
                index = word_to_idx[word]
                word_counts[index] += 1
        return word_counts.reshape(1, -1)
    except Exception as e:
        print(f"Word counting error: {str(e)}")
        return np.zeros((1, len(words)))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    try:
        if request.method == 'POST':
            message = request.form.get('message', '').strip()
            if message:
                tokens = preprocess_text(message)
                if tokens:
                    word_counts = text_to_word_counts(tokens)
                    pred = nb_classifier.predict(word_counts)[0]
                    prediction = "Spam" if pred == 1 else "Not Spam"
                else:
                    prediction = "Error processing message"
            else:
                prediction = "Please enter a message"
    except Exception as e:
        prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)