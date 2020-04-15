import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import joblib
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load data from files
data_rock = joblib.load('data/rock.pkl')
data_paper = joblib.load('data/paper.pkl')
data_scissors = joblib.load('data/scissors.pkl')

# preparing DataFrame
df1 = pd.DataFrame(data_rock)      # 1
df2 = pd.DataFrame(data_paper)     # 2
df3 = pd.DataFrame(data_scissors)  # 3
df = pd.concat([df1, df2, df3])
df = df.drop(['size'], axis=1)
mg_values = df['data'].values
X = np.array([np.asarray(i) for i in mg_values])
y = df['label'].values
X = X.reshape(81, -1)

plt.imshow(mg_values[45])
plt.show()
rock_test = mg_values[45]
rock_test = np.asarray(rock_test)
rock_test = rock_test.reshape(1, -1)

# Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=35, shuffle=True)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Model score: ',  model.score(X_test, y_test))
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))

print(model.predict(rock_test))
