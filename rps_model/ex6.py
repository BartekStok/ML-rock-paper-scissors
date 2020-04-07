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
y = df['Survived'].values
kf = KFold(n_splits=5, shuffle=True)

splits = list(kf.split(X))
score = []
for split in splits:
    train_indices, test_indices = split
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score.append(model.score(X_test, y_test))

print(score)
print(np.mean(score))

final_model = LogisticRegression()
final_model.fit(X, y)

# TODO model evaluation based on different df entry

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)[:, 1]
# fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
# plt.plot(fpr, tpr)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('1 - specificity')
# plt.ylabel('sensitivity')
# plt.show()
#
# print(roc_auc_score(y_test, y_pred_proba))
#
# print(model.score(X_test, y_test))
# print(precision_score(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(recall_score(y_test, y_pred))
# print(f1_score(y_test, y_pred))
#
#
