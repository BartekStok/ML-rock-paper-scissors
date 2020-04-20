import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/cv_result.csv')
df = df.drop(df.keys()[0], axis=1)


# DataFrame with means
def df_mean():
    df_mean = pd.DataFrame()
    for key in df.keys():
        values = []
        for idx, val in df[key].items():
            val = eval(val)
            val = np.mean(val)
            values.append(val)
        df_mean[key] = values
    return df_mean


# DataFrame with all scores
def df_all():
    df_all = pd.DataFrame()
    for key in df.keys():
        values = []
        for idx, val in df[key].items():
            val = eval(val)
            values.append(val)
        df_all[key] = values
    return df_all


df_all = df_all()
df_mean = df_mean()


def clf_comparison1():
    """
    Plots comparison between classifiers scores, using boxplot
    """
    labels = np.array([key for key in df_all.keys()])
    scores = ['Accuracy', 'Precision', 'Recall', 'F1 score']
    fig, axis = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
    for idx, (ax, score) in enumerate(zip(axis, scores)):
        bplot = ax.boxplot(df_all.values[idx, ], showmeans=True, meanline=True)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel(score, fontsize=20)
    fig.tight_layout()
    fig.show()


def clf_comparison2():
    labels = np.array([key for key in df_mean.keys()])
    scores = ['Accuracy', 'Precision', 'Recall', 'F1 score']
    bar_count = np.arange(labels.size)
    bar_height = df_mean.values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(bar_count)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0, (np.max([bar_height]) + 0.1))
    ax.set_ylabel('Value = Precision')
    for idx, score in enumerate(scores):
        bar = ax.bar(bar_count, bar_height[idx, ], label=score)
        for b in bar:
            if score == 'Precision':
                ax.annotate(f'{round(b.get_height(), 2)}',
                            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 2),
                            textcoords='offset points',
                            ha='center',
                            va='bottom'
                            )
    ax.legend(loc=(0.25, 0.8))
    fig.show()


#Plotting
clf_comparison1()
clf_comparison2()
