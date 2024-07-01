import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.datasets import load_iris

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

def manhattan_distance(p, q):
    return np.sum(np.absolute(np.array(p) - np.array(q)))

def chebyshev_distance(p, q):
    return np.max(np.absolute(np.array(p) - np.array(q)))

def calculate_distance(metric, p, q):
    if metric == "manhattan":
        return manhattan_distance(p, q)
    elif metric == "chebyshev":
        return chebyshev_distance(p, q)
    return euclidean_distance(p, q)

class KNearestNeighbors(object):
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y, k=3, metric="manhattan"):
        self.k = k
        self.metric = metric
        self.points = self.encoder.fit_transform(X)
        self.labels = y

    def predict(self, new_points):
        prediction = []
        
        for new_point in new_points:
            distances = np.array([])
            encoded_point = self.encoder.transform([new_point])
    
            for point in self.points:
                distance = calculate_distance(self.metric, point, encoded_point[0])
                distances = np.append(distances, distance)
    
            min_indexes = distances.argsort()[:self.k]
    
            labels_count = {}
            for index in min_indexes:
                if not self.labels[index] in labels_count.keys():
                    labels_count[self.labels[index]] = 1
                else:
                    labels_count[self.labels[index]] += 1

            prediction.append(max(labels_count, key=labels_count.get))
        return prediction

    def get_params(self, deep=False):
        return {}

def cross_validation(df, clf, label_field):
    clone_classifier = clone(clf)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=77)

    y_train = df_train[label_field].to_numpy()
    X_train = df_train.drop(label_field, axis=1).to_numpy()
    
    y_test = df_test[label_field].to_numpy()
    X_test = df_test.drop(label_field, axis=1).to_numpy()
    clone_classifier.fit(X_train, y_train)

    labels_predict = clone_classifier.predict(X_test)
    n_correct = sum(labels_predict == y_test)
    return n_correct / len(labels_predict)

# Завантаження датасету Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Розподіл даних на ознаки і цільову змінну
y_iris = data['target']
x_iris = data.drop("target", axis=1)

clf_iris = KNearestNeighbors()
clf_iris.fit(x_iris, y_iris)

# Тестування класифікатора
prediction = clf_iris.predict([
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5],
    [5.9, 3.0, 5.1, 1.8]
])

print(f"Prediction = {prediction}")

cross_val = cross_validation(data, clf_iris, "target")
print(f"Cross-validation Accuracy = {cross_val}")
