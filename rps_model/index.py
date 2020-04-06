import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

pd.options.display.max_columns = 7

df = pd.read_csv('titanic.csv')
small_df = df[['Age', 'Fare']]
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# X = df[['Fare', 'Age']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
print(model.predict(X))

# plt.scatter(X.min(), y, c=df['Survived'])
# plt.scatter(X, y, c=df['Survived'])
# plt.plot(model.coef_, model.intercept_)
# plt.show()




# text = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
# text.to_csv('titanic.csv')

# col = df['Fare']
# small_df = df[['Age', 'Sex', 'Survived']]
#
# with open('../passangers-titanic.csv', 'w') as f:
#     f.write(df.to_string())


# print(df.head())
# print(col)
# print(small_df)
