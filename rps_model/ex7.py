import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold

pd.options.display.max_columns = 6

df = pd.read_csv('titanic.csv')

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X_age = df[['Age']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X_age, y)
# print(model.predict(X_age))
# print(X_age.shape)
print((model.predict_proba(X_age)[:, 1] > 0.75).sum())
print(np.mean(X_age))
print(np.max(X_age))
print(np.min(X_age))
print(np.median(X_age))
plt.scatter(X_age, X_age[::-1], c=y)
plt.show()
