from BayesClassifier import BayesClassifier
from KNNClassifier import KNNClassifier
import pandas as pd
def BayesPrinter():
    # Running BayesClassifier
    bayes_classifier = BayesClassifier()
    bayes_classifier.evaluate_train_and_test()
def KNNPrinter():
    # Running KNNClassifier
    knn_classifier = KNNClassifier(k=5)
    data = pd.read_csv('spam_dataset.csv')
    X = data['text']
    y = data['label']
    print("Bayes")
    knn_classifier.fit(X, y)
    knn_classifier.predict()
def main():
    BayesPrinter()
    KNNPrinter()

if __name__ == "__main__":
    main()
