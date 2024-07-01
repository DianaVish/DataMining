import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import clone

class OneR(object):
    
    def __init__(self):
        self.ideal_feature = None
        self.ideal_feature_index = 0
        self.max_accuracy = 0
        self.result = dict()
    
    def fit(self, X, y):
        response = list()
        
        dfx = pd.DataFrame(X)
        
        for feature_index, feature in enumerate(dfx):
            self.result[str(feature)] = dict()
            options_values = set(dfx[feature])
            join_data = pd.DataFrame({"variable":dfx[feature], "label":y})
            cross_table = pd.crosstab(join_data.variable, join_data.label)   

            summary = cross_table.idxmax(axis=1)
            self.result[str(feature)] = dict(summary)
    
            correct_answers = 0
            for idx, row in join_data.iterrows():
                if row['label'] == self.result[str(feature)][row['variable']]:
                    correct_answers += 1

            accuracy = (correct_answers/len(y))
            
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.ideal_feature = feature
                self.ideal_feature_index = feature_index

            result_feature = {"feature": str(feature), "accuracy":accuracy, "rules": self.result[str(feature)] } 
            print(result_feature)
            response.append(result_feature)
            
        return response

    def predict(self, X):
        predict_result = []

        print(self.result)

        for item in X:
            value = self.result[str(self.ideal_feature)].get(item[self.ideal_feature_index], None)
            predict_result.append(value)

        return predict_result

    def get_params(self, deep = False):
        return {}
           
    def __repr__(self):
        if self.ideal_feature != None:
            message = "Most accurate feature is: " + str(self.ideal_feature)
        else:
            message = "Cannot choose most accurate feature"
        return message

# Завантаження датасету Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Розподіл даних на ознаки і цільову змінну
y = data['target']
X = data.drop("target", axis=1)

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

# Ініціалізація та навчання класифікатора OneR
clf_iris = OneR()
results = clf_iris.fit(X, y)

print(results)
print(clf_iris)

# Тестування класифікатора
predicted_data = clf_iris.predict(
    [[5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5],
    [5.9, 3.0, 5.1, 1.8]]
)
print(f"Predict = {predicted_data}")

cross_val = cross_validation(data, clf_iris)
print(f"Accuracy = {cross_val}")

