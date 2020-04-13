import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = 7

df = pd.read_csv('titanic.csv')
small_df = df[['Age', 'Fare']]
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# X = df[['Fare', 'Age']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
print((y == y_pred).sum() / y.shape[0])
print(model.score(X, y))

print(precision_score(y, y_pred))
print(accuracy_score(y, y_pred))
print(recall_score(y, y_pred))
print(f1_score(y, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=35)

model_tr = LogisticRegression()
model_tr.fit(X_train, y_train)
print(model_tr.score(X_test, y_test))


# plt.scatter(df['Fare'], df['Age'], c=df['Survived'])
# plt.scatter(X, y, c=df['Survived'])
# plt.plot()
# plt.show()

# text.to_csv('titanic.csv')

# col = df['Fare']
# small_df = df[['Age', 'Sex', 'Survived']]
#
# with open('../passangers-titanic.csv', 'w') as f:
#     f.write(df.to_string())


# print(df.head())
# print(col)
# print(small_df)
