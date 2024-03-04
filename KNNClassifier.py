import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


class KNNClassifier:
    def __init__(self, file_path, k=5):
        self.file_path = file_path
        self.k = k
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self):
        # Load the dataset
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self, test_size=0.2, random_state=42):
        # Separate features and target variable
        X = self.data.drop(columns=['fetal_health'])
        y = self.data['fetal_health']

        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_imputed, y, test_size=test_size,
                                                                                random_state=random_state)

        # Reset indices of y_train
        self.y_train = self.y_train.reset_index(drop=True)

    def euclidean_distance(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)

        # Ensure x1 and x2 have compatible shapes
        if x1.shape != x2.shape:
            raise ValueError("Not the same.")

        # Compute the Euclidean distance
        return np.linalg.norm(x1 - x2)

    def fit(self):
        pass  # KNN does not require explicit training step

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Convert the input data to numpy array
        x = np.array(x)
        # Reshape the input data to ensure it's in the right format
        x_reshaped = x.reshape(1, -1)
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x_reshaped, x_train.reshape(1, -1)) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def evaluate(self, X_train, y_train):
        # Convert input features to numpy array
        X_train = np.array(X_train)

        # Make predictions on training set
        y_train_pred = self.predict(X_train)

        # Convert y_train to a numpy array if it's a pandas series
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Handle NaN values in y_true
        y_train = np.nan_to_num(y_train, nan=np.nanmedian(y_train))  # Replace NaN values with the median

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_train, y_train_pred)
        report = classification_report(y_train, y_train_pred, output_dict=True)

        # Print results for training set
        print("Accuracy:", accuracy)
        print("Recall:", report['macro avg']['recall'])
        print("Precision:", report['macro avg']['precision'])
        print("F1-score:", report['macro avg']['f1-score'])
