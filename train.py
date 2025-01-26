import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

def train_model():
    if not os.path.exists('emails.csv'):
        raise FileNotFoundError("emails.csv file not found in the current directory")

    try:
        df = pd.read_csv('emails.csv')
        
        if df.shape[1] != 3002:
            raise ValueError(f"Expected 3002 columns but got {df.shape[1]}")

        num_word_columns = 3000
        column_names = ['Email_ID'] + [f'word_{i}' for i in range(1, num_word_columns + 1)] + ['Label']
        df.columns = column_names

        X = df.iloc[:, 1:-1]
        y = df['Label']
        X = X.fillna(0).astype(int)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        model_filename = 'nb_classifier.pkl'
        with open(model_filename, 'wb') as model_file:
            pickle.dump(nb_classifier, model_file)
        print(f"\nModel saved as {model_filename}")
        
        return nb_classifier

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    train_model()
