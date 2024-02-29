import pandas as pd
from KNNClassifier import KNNClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
from KNNClassifier import KNNClassifier
from sklearn.model_selection import train_test_split

def main():
    # Load the dataset
    file_path = "fetal_health.csv"
    knn_classifier = KNNClassifier(file_path=file_path, k=5)  # Set the value of k here

    # Load and preprocess the data
    data = knn_classifier.load_dataset()
    X, y = knn_classifier.preprocess_data(data)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the kNN classifier
    knn_classifier.fit(X_train, y_train)

    # Evaluate the classifier
    knn_classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
