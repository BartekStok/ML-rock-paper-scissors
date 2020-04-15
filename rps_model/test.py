import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import joblib
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image


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
X = df['data'].values
y = df['label'].values

# print(np.asarray(df1['data'][0]))
test = np.asarray(df1['data'][0])
arr = np.empty((336, 336, 3), int)

for i in range(len(df1['data'])):
    temp_arr = np.asarray(df1['data'][i])
    np.concatenate((arr, temp_arr), axis=0)

# print(arr)
print(test)