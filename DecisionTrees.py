import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def cross_validation(X, y, clf):
    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    predictions = []

    for train_index, test_index in skfolds.split(X, y):
        clone_classifier = clone(clf)
        
        X_train = X[train_index]
        y_train = y[train_index]
        
        X_test = X[test_index]
        y_test = y[test_index]
    
        clone_classifier.fit(X_train, y_train)
        labels_predict = clone_classifier.predict(X_test)
        n_correct = sum(labels_predict == y_test)
        predictions.append(n_correct / len(labels_predict))
        
    return predictions

# Завантаження датасету Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Розподіл даних на ознаки і цільову змінну
y_iris = data['target']
x_iris = data.drop("target", axis=1)

# Ініціалізація класифікатора Decision Tree
clf = DecisionTreeClassifier(random_state=42)

# Закодуємо дані за допомогою OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = encoder.fit_transform(x_iris)

# Навчання моделі
clf = clf.fit(X , y_iris)

# Візуалізація дерева рішень
tree.plot_tree(clf)

# Виконання крос-валідації
predictions = cross_validation(X, y_iris, clf)
print(f"Cross-validation predictions: {predictions}")
print(f"Average accuracy: {sum(predictions)/len(predictions)}")
