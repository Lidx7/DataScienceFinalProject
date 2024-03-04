import random
import math
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class BayesClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def loadData(self):
        self.data = pd.read_csv(self.file_path)


    def preprocessData(self):
        # Separate features and target variable
        X = self.data.drop(columns=['fetal_health'])
        y = self.data['fetal_health']

        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

    def splitData(self, ratio):
        train_num = int(len(self) * ratio)
        train = []
        test = list(self)
        while len(train) < train_num:
            index = random.randrange(len(test))
            train.append(test.pop(index))
        return train, test

    def groupUnderClass(self):
        data_dict = {}
        for i in range(len(self)):
            if self[i][-1] not in data_dict:
                data_dict[self[i][-1]] = []
            data_dict[self[i][-1]].append(self[i])
        return data_dict

    def MeanAndStdDev(self):
        avg = np.mean(self)
        stddev = np.std(self)
        return avg, stddev

    def MeanAndStdDevForClass(self):
        info = {}
        data_dict = self.groupUnderClass(self)
        for classValue, instances in data_dict.items():
            info[classValue] = [self.MeanAndStdDev(attribute) for attribute in zip(*instances)]
        return info

    def calculateGaussianProbability(x, mean, stdev):
        epsilon = 1e-10
        expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
        return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

    def calculateClassProbabilities(info, test):
        probabilities = {}
        for classValue, classSummaries in info.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, std_dev = classSummaries[i]
                x = test[i]
                probabilities[classValue] *= info.calculateGaussianProbability(x, mean, std_dev)
        return probabilities

    def predict(info, test):
        probabilities = info.calculateClassProbabilities(info, test)
        bestLabel = max(probabilities, key=probabilities.get)
        return bestLabel

    def getPredictions(info, test):
        predictions = [info.predict(info, instance) for instance in test]
        return predictions

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


