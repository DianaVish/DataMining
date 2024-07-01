import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.datasets import load_iris
import numpy as np

class GaussianNaiveBayes:
    
    def __init__(self):
        self.model = {}
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.model = {c: {} for c in self.classes}
        for c in self.classes:
            X_c = X[y == c]
            self.model[c]['mean'] = X_c.mean(axis=0)
            self.model[c]['var'] = X_c.var(axis=0)
            self.model[c]['prior'] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.model[c]['prior'])
            posterior = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self.model[class_idx]['mean']
        var = self.model[class_idx]['var']
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def get_params(self, deep=False):
        return {}

def cross_validation(df, clf):
    clone_classifier = clone(clf)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=77)

    y_train = df_train["target"].to_numpy()
    X_train = df_train.drop("target", axis=1).to_numpy()
    
    y_test = df_test["target"].to_numpy()
    X_test = df_test.drop("target", axis=1).to_numpy()
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

clf_naive_bayes = GaussianNaiveBayes()
results = clf_naive_bayes.fit(x_iris.to_numpy(), y_iris.to_numpy())

cross_val = cross_validation(data, clf_naive_bayes)
print(f"Accuracy = {cross_val}")
