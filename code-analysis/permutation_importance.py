import os,sys
sys.path.append('../')

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy

from src.runner import Runner
from src.model_MLP import ModelMLP

sns.set()

def permuted(matrix, columns):
    """特定のカラムをシャッフルしたデータフレームを返す"""
    for column_name in columns:
        if column_name == "mfcc":
            l,r = 0, 39
        elif column_name == "delta":
            l,r = 39, 78
        elif column_name == "power":
            l,r = 78, 849
        permuted_mat = deepcopy(matrix)
        L = len(permuted_mat)
        permuted_mat[:,l:r] = np.array([ np.random.permutation(permuted_mat[i, l:r]) for i in range(L) ] )
        yield column_name, permuted_mat

def pimp(clf, X, y, columns, cv=None, eval_func=accuracy_score):
    """PIMP (Permutation IMPortance) を計算する"""
    base_scores = []
    permuted_scores = defaultdict(list)

    if cv is None:
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=71)

    for train_index, test_index in cv.split(X, y):
        # 学習用データと検証用データに分割する
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        # 学習用データでモデルを学習する
        clf.fit(X_train, y_train, X_test, y_test)

        # まずは何もシャッフルしていないときのスコアを計算する
        y_pred_base = np.argmax(clf.predict(X_test),axis=1)
        base_score = eval_func(y_test, y_pred_base)
        base_scores.append(base_score)

        # 特定のカラムをシャッフルした状態で推論したときのスコアを計算する
        permuted_X_test_gen = permuted(X_test, columns)
        for column_name, permuted_X_test in permuted_X_test_gen:
            y_pred_permuted = np.argmax(clf.predict(permuted_X_test),axis=1)
            permuted_score = eval_func(y_test, y_pred_permuted)
            permuted_scores[column_name].append(permuted_score)

    # 基本のスコアとシャッフルしたときのスコアを返す
    np_base_score = np.array(base_scores)
    dict_permuted_score = {name: np.array(scores) for name, scores in permuted_scores.items()}
    return np_base_score, dict_permuted_score

def score_difference_statistics(base, permuted):
    """シャッフルしたときのスコアに関する統計量 (平均・標準偏差) を返す"""
    mean_base_score = base.mean()
    for column_name, scores in permuted.items():
        score_differences = scores - mean_base_score
        yield column_name, score_differences.mean(), score_differences.std()



features = ["mfcc", "delta", "power"]
X = Runner.load_x_train(features)
y = Runner.load_y_train()
# 計測に使うモデルを用意する
params = {
        'input_dropout': 0.0,
        'hidden_layers': 3,
        'hidden_units': 96,
        'hidden_activation': 'relu',
        'hidden_dropout': 0.2,
        'batch_norm': 'before_act',
        'optimizer': {'type': 'adam', 'lr': 0.001},
        'batch_size': 64,
        'nb_epoch': 1000,
}
p = {'batch_norm': 'no', 'batch_size': 64.0, 'hidden_activation': 'prelu', 'hidden_dropout': 0.45, 'hidden_layers': 3.0, 'hidden_units': 224.0, 'input_dropout': 0.2, 
    'optimizer': {'lr': 0.00091, 'type': 'adam'}}
params.update(p)
clf = ModelMLP("MLP", **params)

# Permutation Importance を計測する
base_score, permuted_scores = pimp(clf, X, y, features)

# 計測結果から統計量を計算する
diff_stats = list(score_difference_statistics(base_score, permuted_scores))

# カラム名、ベーススコアとの差、95% 信頼区間を取り出す
sorted_diff_stats = sorted(diff_stats, key=lambda x: x[1])
column_names = [name for name, _, _ in sorted_diff_stats]
diff_means = [diff_mean for _, diff_mean, _ in sorted_diff_stats]
diff_stds_95 = [diff_std * 1.96 for _, _, diff_std in sorted_diff_stats]

# グラフにプロットする
plt.plot(column_names, diff_means, marker='o', color='r', markersize=10)
plt.errorbar(column_names, diff_means, yerr=diff_stds_95, ecolor='g', capsize=4)

plt.title('Permutation Importance')
plt.grid()
plt.xlabel('column')
plt.ylabel('difference')
plt.plot()
os.makedirs('./fig', exist_ok=True)
plt.savefig('./fig/permutation.png',dpi=300)
