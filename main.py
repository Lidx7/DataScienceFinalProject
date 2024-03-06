import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from BayesClassifier import BayesClassifier
from KNNClassifier import KNNClassifier

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def BayesPrinter():
    # Running BayesClassifier
    bayes_classifier = BayesClassifier()
    bayes_classifier.evaluate_train_and_test()

    # Calculate confusion matrix for training set
    cm_train = confusion_matrix(bayes_classifier.y_train, bayes_classifier.predict(bayes_classifier.X_train_vectorized))
    # Visualize confusion matrix for training set
    plot_confusion_matrix(cm_train, classes=['Not Spam', 'Spam'], title='Confusion Matrix - Bayes Classifier (Training)')

    # Calculate confusion matrix for testing set
    cm_test = confusion_matrix(bayes_classifier.y_test, bayes_classifier.predict(bayes_classifier.X_test_vectorized))
    # Visualize confusion matrix for testing set
    plot_confusion_matrix(cm_test, classes=['Not Spam', 'Spam'], title='Confusion Matrix - Bayes Classifier (Testing)')

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

    # Calculate confusion matrix for training set
    cm_train = confusion_matrix(knn_classifier.y_train, knn_classifier.knn_classifier.predict(knn_classifier.X_train))
    # Visualize confusion matrix for training set
    plot_confusion_matrix(cm_train, classes=['Not Spam', 'Spam'], title='Confusion Matrix - KNN Classifier (Training)')

    # Calculate confusion matrix for testing set
    cm_test = confusion_matrix(knn_classifier.y_test, knn_classifier.knn_classifier.predict(knn_classifier.X_test))
    # Visualize confusion matrix for testing set
    plot_confusion_matrix(cm_test, classes=['Not Spam', 'Spam'], title='Confusion Matrix - KNN Classifier (Testing)')

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
    GeneralPrints()
    BayesPrinter()
    KNNPrinter()

if __name__ == "__main__":
    main()
