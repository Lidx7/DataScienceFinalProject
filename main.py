import sns
from matplotlib import pyplot as plt
from BayesClassifier import BayesClassifier
from KNNClassifier import KNNClassifier
import pandas as pd
import seaborn as sns
def BayesPrinter():
    # Running BayesClassifier
    bayes_classifier = BayesClassifier()
    bayes_classifier.evaluate_train_and_test()
def KNNPrinter():
    k=3
    # Running KNNClassifier
    knn_classifier = KNNClassifier(k=k)
    data = pd.read_csv('spam_dataset.csv')
    X = data['text']
    y = data['label']
    print("KNN classifier for k=",k ,"\n")
    knn_classifier.fit(X, y)
    knn_classifier.predict()
def GeneralPrints():
    df = pd.read_csv("spam_dataset.csv")
    print(df.head(3))
    print("--------------------")
    df.info()
    sns.countplot(x="label", data=df)
    plt.show()
    labels = {0: "Not Spam", 1: "Spam"}
    label_counts = df['label'].value_counts()
    print(label_counts)
    plt.pie(label_counts, labels=labels.values(), autopct="%.2f%%")
    plt.show()

def main():
    # GeneralPrints()
    # BayesPrinter()
    KNNPrinter()


if __name__ == "__main__":
    main()