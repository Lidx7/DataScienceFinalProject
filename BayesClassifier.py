import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BayesClassifier:
    def __init__(self):
        # Load the dataset
        self.data = pd.read_csv("spam_dataset.csv")

        # Separate features and labels
        self.X = self.data['text']
        self.y = self.data['label']

        # Splitting the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Vectorizing the text data
        self.vectorizer = CountVectorizer()
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)

        # Training the Naive Bayes Classifier
        self.nb_classifier = MultinomialNB()
        self.nb_classifier.fit(self.X_train_vectorized, self.y_train)

    def predict(self, X):
        return self.nb_classifier.predict(X)

    def evaluate(self, y_true, y_pred, dataset_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"{dataset_name} Set Metrics:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print()

    def evaluate_train_and_test(self):
        # Predictions for training set
        y_train_pred = self.predict(self.X_train_vectorized)

        # Predictions for testing set
        y_test_pred = self.predict(self.X_test_vectorized)

        # Print evaluation metrics for both training and testing sets
        self.evaluate(self.y_train, y_train_pred, "Training")
        self.evaluate(self.y_test, y_test_pred, "Testing")