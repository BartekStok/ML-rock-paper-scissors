import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import joblib
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection._split import KFold


# load data from files
data_rock = joblib.load('data/rock.pkl')
data_paper = joblib.load('data/paper.pkl')
data_scissors = joblib.load('data/scissors.pkl')

# TODO function to load data, concatenate, return df, in utils.py
# preparing DataFrame
df1 = pd.DataFrame(data_rock)      # 1
df2 = pd.DataFrame(data_paper)     # 2
df3 = pd.DataFrame(data_scissors)  # 3
df = pd.concat([df1, df2, df3])
# df = df.drop(['size'], axis=1)
img_values = df['data'].values
img_array = np.array([np.asarray(i) for i in img_values])
X = img_array.reshape(img_values.size, -1)
y = df['label'].values

# Plotting sample pictures
labels = np.unique(y)
fig, axes = plt.subplots(1, labels.size)
fig.tight_layout()
fig.suptitle('Random pictures example', fontsize=25)
for ax, label in zip(axes, labels):
    idx = np.random.choice(np.argwhere(y == label).reshape(-1))
    ax.imshow(img_values[idx])
    ax.set_title(y[idx], fontsize=18)
    ax.axis('off')
fig.show()

# Plotting number of each samples in sample train and test split
plot_train_X, plot_test_X, plot_train_y, plot_test_y = train_test_split(
    X,
    y,
    shuffle=True,
    random_state=36
)
fig2, ax = plt.subplots(1)
fig2.show()

# Cross Validation
# kf = KFold(n_splits=6, shuffle=True)
# splits = kf.split(X)
#
# model_accuracy = []
# model_precision = []
# model_recall = []
# model_f1score = []
# for split in splits:
#     train_indices, test_indices = split
#     X_train = X[train_indices]
#     X_test = X[test_indices]
#     y_train = y[train_indices]
#     y_test = y[test_indices]
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)
#     model_accuracy.append(accuracy_score(y_test, y_pred))
#     model_precision.append(precision_score(y_test, y_pred, average='weighted'))
#     model_recall.append(recall_score(y_test, y_pred, average='weighted'))
#     model_f1score.append(f1_score(y_test, y_pred, average='weighted'))
#
# print('Accuracy: ', np.mean(model_accuracy))
# print('Recall: ', np.mean(model_recall))
# print('Precision: ', np.mean(model_precision))
# print('F1 score: ', np.mean(model_f1score))

# Plotting results
# fig, ax = plt.subplots()
# ax.scatter(X_test[:, 0], y_pred_proba[:, 0], label='rock')
# ax.scatter(X_test[:, 0], y_pred_proba[:, 1], label='paper')
# ax.scatter(X_test[:, 0], y_pred_proba[:, 2], label='scissors')
#
# # plt.subplot()
# # plt.plot()
#
# fig.show()

# Training model
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=35, shuffle=True)
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Model score: ',  model.score(X_test, y_test))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred, average='weighted'))
# print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
# print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))

# final_model = RandomForestClassifier()
# final_model.fit(X, y)
# joblib.dump(final_model, 'model.joblib')
