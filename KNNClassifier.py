import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


class KNNClassifier:
    def __init__(self, file_path, k=5):
        self.file_path = file_path
        self.k = k
        self.knn = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def load_dataset(self):
        # Load the dataset
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self, data):
        # Separate features and target variable
        X = data.drop(columns=['fetal_health'])
        y = data['fetal_health']

        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return X_scaled, y_encoded

    def fit(self, X_train, y_train):
        # Initialize and fit the kNN classifier
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions
        return self.knn.predict(X_test)

    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.predict(X_test)

        # Evaluate the algorithm's performance
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Print the evaluation metrics
        print("Accuracy:", accuracy)
        print("Recall:", report['macro avg']['recall'])
        print("Precision:", report['macro avg']['precision'])
        print("F1-score:", report['macro avg']['f1-score'])

