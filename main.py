from KNNClassifier import KNNClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Create KNNClassifier object
    knn_classifier = KNNClassifier(file_path='fetal_health.csv', k=5)

    # Load dataset
    data = knn_classifier.load_data()

    # Preprocess data
    X, y = knn_classifier.preprocess_data(data)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    knn_classifier.fit(X_train, y_train)

    # Evaluate the model
    accuracy, recall, precision, f1_score = knn_classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()

