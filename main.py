import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from KNNClassifier import KNNClassifier


def main():
    KnnClassifier()
    BayesClassifier()


def KnnClassifier():
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


# TODO: fix this load of bullshit. i just copied what was on the KNN classifier function and tried to
#  adjust it somehow to the Bayes classifier function
def BayesClassifier():
    def BayesClassifier():
        # Load the dataset
        file_path = "fetal_health.csv"
        bayes_classifier = GaussianNB()

        # Load and preprocess the data
        bayes_classifier.load_data()
        bayes_classifier.preprocess_data()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        # Train the Bayes classifier (you need to implement this method)
        bayes_classifier.train()

        # Evaluate the classifier's performance
        print("Results on Training Set:")
        train_predictions = bayes_classifier.getPredictions(bayes_classifier.X_train)
        train_accuracy = bayes_classifier.accuracy_rate(bayes_classifier.y_train, train_predictions)
        print("Accuracy on Training Set:", train_accuracy)

        print("\nResults on Test Set:")
        test_predictions = bayes_classifier.getPredictions(bayes_classifier.X_test)
        test_accuracy = bayes_classifier.accuracy_rate(bayes_classifier.y_test, test_predictions)
        print("Accuracy on Test Set:", test_accuracy)


if __name__ == "__main__":
    main()