import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import joblib
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection._split import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from rps_model.settings import LABELS


def load_data_frame():
    """
    Loads data from files and concatenates

    :return: Concatenated Pandas DataFrame
    """
    paths = [f'data/{label}.pkl' for label in LABELS]
    single_df = [pd.DataFrame(joblib.load(path)) for path in paths]
    df_concat = pd.concat(single_df)
    return df_concat


# Load data and set X and y
df = load_data_frame()
img_values = df['data'].values
img_array = np.array([np.asarray(i) for i in img_values])
X = img_array.reshape(img_values.size, -1)
y = df['label'].values


def plot_image():
    """
    PLots sample images with labels
    """

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


def plot_amount():
    """
    Plotting amount of each sample divided for (train and test) split
    """

    # Splitting X and y, preparing data
    plot_train_X, plot_test_X, plot_train_y, plot_test_y = train_test_split(
        X,
        y,
        shuffle=True,
        # random_state=36
    )
    labels = np.unique(y)
    bar_count = np.arange(labels.size)
    amounts_train = [(plot_train_y == label).sum() for label in labels]
    amounts_test = [(plot_test_y == label).sum() for label in labels]
    width = 0.35
    # Setting up plot space
    fig, ax = plt.subplots()
    ax.set_xlabel('Labels', fontsize=16)
    ax.set_ylabel('Amount', fontsize=16)
    ax.set_title('Amount of images per type', fontsize=16)
    ax.set_xticks(bar_count)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, (np.max([amounts_train]) + 10))
    # Plotting each bar with given data
    bar1 = ax.bar(bar_count - width / 2,
                  amounts_train,
                  width,
                  label=f'Train - {sum(amounts_train)} images'
                  )
    bar2 = ax.bar(bar_count + width / 2,
                  amounts_test,
                  width,
                  label=f'Test - {sum(amounts_test)} images'
                  )
    # Adding number of samples over bars
    for b1, b2 in zip(bar1, bar2):
        ax.annotate(f'{b1.get_height()}',
                    xy=(b1.get_x() + b1.get_width() / 2, b1.get_height()),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha='center',
                    va='bottom'
                    )
        ax.annotate(f'{b2.get_height()}',
                    xy=(b2.get_x() + b2.get_width() / 2, b2.get_height()),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha='center',
                    va='bottom'
                    )
    ax.legend()
    fig.show()


# Plotting
plot_image()
plot_amount()


def cross_validation():
    """
    Computes multiple classifiers, saves scores, plots data file

    :return: Dict with classifiers scores
    """
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        SGDClassifier()
    ]
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "SGD"
    ]
    # Splitting data for cross validation
    kf = KFold(n_splits=5, shuffle=True)
    splits = list(kf.split(X))
    cv_result = {}
    # Iterates trough classifiers, saving each score to a dict
    for classifier, name in zip(classifiers, names):
        model = classifier
        model_accuracy = []
        model_precision = []
        model_recall = []
        model_f1score = []
        for split in splits:
            train_indices, test_indices = split
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_accuracy.append(accuracy_score(y_test, y_pred))
            model_precision.append(precision_score(y_test, y_pred, average='weighted'))
            model_recall.append(recall_score(y_test, y_pred, average='weighted'))
            model_f1score.append(f1_score(y_test, y_pred, average='weighted'))
        cv_result[name] = [model_accuracy, model_precision, model_recall, model_f1score]
    # Saving to a data file
    export_df = pd.DataFrame()
    for key, values in cv_result.items():
        export_df[f'{key}'] = values
    export_df.to_csv('data/cv_result.csv')

    return cv_result


# cv_result = cross_validation()


# Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=36, shuffle=True)
model = SGDClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Model score: ', model.score(X_test, y_test))
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))

# final_model = RandomForestClassifier()
# final_model = SGDClassifier()
# final_model.fit(X, y)
# joblib.dump(final_model, 'model.joblib')
