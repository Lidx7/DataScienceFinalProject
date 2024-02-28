import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
class KNNClassifier:
    def __init__(self, file_path, k=5):
        self.file_path = file_path
        self.k = k
        self.knn = None
        self.scaler = StandardScaler()
    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data
    def preprocess_data(self,data):
        x = data.drop('fetal_health', axis =1)
        y = data['fetal_health']
        X_scaled = self.scaler.fit_transform(x)
        return X_scaled,y
    def fit(self, X_train, y_train):
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(X_train, y_train)
    def predict(self,X_test):
        return self.knn.predict(X_test)
    def evaluate(self,X_test,y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        # Print and return accuracy, recall, precision, and f1-score
        print("Accuracy:", accuracy)
        print("Recall:", report['macro avg']['recall'])
        print("Precision:", report['macro avg']['precision'])
        print("F1-score:", report['macro avg']['f1-score'])
        return accuracy, report['macro avg']['recall'], report['macro avg']['precision'], report['macro avg']['f1-score']