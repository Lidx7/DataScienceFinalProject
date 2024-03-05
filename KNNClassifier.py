import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class KNNClassifier():
    # Load dataset
    data = pd.read_csv('spam_dataset.csv')

    # Splitting data into features and labels
    X = data['text']
    y = data['label']

    # Convert text data into numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Naive Bayes classifier
    nb_classifier = MultinomialNB()

    # Train the classifier
    nb_classifier.fit(X_train, y_train)

    # Predictions for training set
    y_train_pred = nb_classifier.predict(X_train)

    # Predictions for testing set
    y_test_pred = nb_classifier.predict(X_test)

    # Evaluation for training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    # Evaluation for testing set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Print evaluation metrics for training set
    print("Training Set Metrics:")
    print("Accuracy:", train_accuracy)
    print("Precision:", train_precision)
    print("Recall:", train_recall)
    print("F1-score:", train_f1)
    print()

    # Print evaluation metrics for testing set
    print("Testing Set Metrics:")
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1-score:", test_f1)
