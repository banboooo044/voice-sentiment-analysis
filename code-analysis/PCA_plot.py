# Dataset1 mfccの主成分分析
import os,sys
sys.path.append('../')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from src.runner import Runner

X = Runner.load_x_train(["mfcc"])
print(X.shape)

#標準化
scaler = StandardScaler()
scaler.fit(X)
standard_X = scaler.transform(X)

# 12次元 -> 6次元 へ落とす.
dim=6
params = {
    'n_components' : dim,
    'random_state' : 71,
}
# 主成分分析
clf = PCA(**params)
clf.fit(standard_X)
pca = clf.transform(standard_X)

pca = pca[:100]
labels = Runner.load_y_train()
labels = labels[:100]

happy = np.empty((0, dim))
angry = np.empty((0, dim))
sad = np.empty((0, dim))

#0=sad, 1=angry, 2=happy, 3=normal
for label, x in zip(labels, pca):
    if label == 0:
        sad = np.vstack((sad, x))
    elif label == 1:
        angry = np.vstack((angry, x))
    elif label == 2:
        happy = np.vstack((happy, x))

## プロット1
fig = plt.figure(figsize=(10, 10))
# 第2主成分ベクトルと第3主成分ベクトルの軸でプロットを描く.
plt.scatter(happy[:, 1], happy[ :,2], marker='D', c="fuchsia", linewidths=4)
plt.scatter(angry[:, 1], angry[:, 2], marker='o', c="#ff2222", linewidths=5)

# 保存
os.makedirs('./fig', exist_ok=True)
plt.savefig('./fig/happy-angry.png',dpi=300)

## プロット2
fig = plt.figure(figsize=(10, 10))
# 第4主成分ベクトルと第5主成分ベクトルの軸でプロットを描く.
plt.scatter(happy[:, 3], happy[ :,4], marker='D', c="fuchsia", linewidths=4)
plt.scatter(sad[:, 3], sad[:, 4], marker='^', c="blue", linewidths=5)

# 保存
plt.savefig('./fig/happy-sad.png',dpi=300)