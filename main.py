from BayesClassifier import BayesClassifier
from KNNClassifier import KNNClassifier
import pandas as pd

def main():
    # Running BayesClassifier
    bayes_classifier = BayesClassifier()
    bayes_classifier.evaluate_train_and_test()

    # Running KNNClassifier
    knn_classifier = KNNClassifier(k=5)
    data = pd.read_csv('spam_dataset.csv')
    X = data['text']
    y = data['label']
    print("Bayes")
    knn_classifier.fit(X, y)
    knn_classifier.predict()

if __name__ == "__main__":
    main()
