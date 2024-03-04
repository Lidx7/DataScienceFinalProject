import pandas as pd
from sklearn.model_selection import train_test_split

from KNNClassifier import KNNClassifier


def main():
    # Load the dataset
    file_path = "fetal_health.csv"
    knn_classifier = KNNClassifier(file_path=file_path, k=20)

    # Load and preprocess the data
    knn_classifier.load_dataset()
    knn_classifier.preprocess_data()

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(knn_classifier.data.drop(columns=['fetal_health']),
                                                        knn_classifier.data['fetal_health'],
                                                        test_size=0.2, random_state=42)
    # df = pd.read_csv(file_path)
    # print(df.info)
    # print(df.describe(),2)
    # Train the kNN classifier
    knn_classifier.fit()
    print("Results on Training Set:")
    knn_classifier.evaluate(X_train, y_train)
    print("\nResults on Test Set:")
    knn_classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()