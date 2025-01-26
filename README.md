# Bayesian Email Spam Classifier

A web-based email spam classifier using Naive Bayes algorithm. This project demonstrates the practical application of Bayesian probability in classifying emails as spam or non-spam.

## Features

- Web interface for easy email classification
- Real-time prediction using Naive Bayes algorithm
- Responsive design with explanation of the classification process
- High accuracy in spam detection

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bayesian-spam-classifier
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the email dataset (emails.csv) and place it in the project root directory. Dataset can be download from the kaggle.com [here](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data).

## Training the Model

1. Extract word features:
```bash
python extract_words.py
```

2. Train the classifier:
```bash
python train.py
```

This will create two files:
- `words.pkl`: Contains the word features
- `nb_classifier.pkl`: The trained model

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Access the web interface through your browser
2. Enter the email text you want to classify
3. Click the "Classify" button
4. View the prediction result (Spam/Not Spam)