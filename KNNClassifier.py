import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.vectorizer = TfidfVectorizer()

    def fit(self, X, y):
        X_vectorized = self.vectorizer.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.k, metric = 'minkowski')
        self.knn_classifier.fit(self.X_train, self.y_train)

    def predict(self):
        y_train_pred = self.knn_classifier.predict(self.X_train)
        y_test_pred = self.knn_classifier.predict(self.X_test)
        self.print_evaluation_metrics(self.y_train, y_train_pred, "Training")
        self.print_evaluation_metrics(self.y_test, y_test_pred, "Testing")

    def print_evaluation_metrics(self, y_true, y_pred, dataset_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"{dataset_name} Set Metrics:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print()