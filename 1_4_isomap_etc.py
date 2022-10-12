import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

# LLE
# with sklearn

from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=11, random_state=42)
lle.fit(X)

LocallyLinearEmbedding(eigen_solver='auto', hessian_tol=0.0001, max_iter=100,method='standard', modified_tol=1e-12, n_components=2,
                       n_jobs=1, n_neighbors=10, neighbors_algorithm='auto', random_state=42, reg=0.001, tol=1e-06)

X_reduced = lle.transform(X)

plt.title('LLE results with sklearn', fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.savefig('1_4_LLE_results_with sklearn')


# t-SNE
# with sklearn

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12,12))
plt.title('t-SNE with sklearn', fontsize=14)
plt.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=y, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.savefig('1_4_t-SNE_results_with sklearn')

