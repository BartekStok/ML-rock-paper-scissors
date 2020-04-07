from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
X = df[cancer_data.feature_names].values
y = df['target'].values
model = LogisticRegression(solver='liblinear')
model.fit(X, y)
y_pred = model.predict(X)
print((y_pred == y).sum() / y.shape[0])
print(model.score(X, y))
print(accuracy_score(y, y_pred))
print('precision score: ',  precision_score(y, y_pred))
print(recall_score(y, y_pred))
print(f1_score(y, y_pred))


print(model.predict([X[45]]), y[45])

