import graphviz as graphviz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

pd.options.display.max_columns = 8

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'
feature_names = ['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
X = df[feature_names].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
model = DecisionTreeClassifier()  # default 'gini'
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(model.predict(X[:15]))  # predicts for first 15 datasets
# print('Model score: ',  model.score(X_test, y_test))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred))
# print('F1 score: ', f1_score(y_test, y_pred))


param_grid = {
    'max_depth': [5, 10, 15, 25, 35],
    'min_samples_leaf': [1, 3, 5],
    'max_leaf_nodes': [5, 10, 20, 35, 50]
}
gs = GridSearchCV(model, param_grid=param_grid, scoring='precision', cv=5)
gs.fit(X, y)
print(gs.best_params_)  # searching fo best params for decision treee, now
                        # we can go back to decision tree, and change params.
print(gs.best_score_)
# dot_file = export_graphviz(model, feature_names=feature_names)
# graph = graphviz.Source(dot_file)
# graph.render(filename='tree_pruned', format='png', cleanup=True)
