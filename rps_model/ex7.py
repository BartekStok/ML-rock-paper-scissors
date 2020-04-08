import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold

pd.options.display.max_columns = 8

df = pd.read_csv('titanic.csv')

df['male'] = df['Sex'] == 'male'
df['AgeSurv'] = (df['Age'].values <= 30) & (df['Survived'].values == 1)
df['SexSurv'] = (df['male'].values == True) & (df['Survived'] == 1)
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X_age = df[['Age']].values > 30
y = df['Survived'].values
print((df['AgeSurv'].values).sum())
print(X_age.sum())
print((df['SexSurv'].values).sum())

plt.scatter(df['Age'], df['Fare'], c=df['Survived'])
plt.show()
