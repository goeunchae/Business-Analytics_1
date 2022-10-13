import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_csv('./data/data_wine.csv')
X = df.drop(['quality'], axis=1)
y = df['quality'].tolist()

# LLE
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=11, random_state=42)
lle.fit(X)

neighbors = [2, 3, 5, 10, 20]
for i in range(len(neighbors)):
    LocallyLinearEmbedding(eigen_solver='auto', hessian_tol=0.0001, max_iter=100,method='standard', modified_tol=1e-12, n_components=2,
                        n_jobs=1, n_neighbors=neighbors[i], neighbors_algorithm='auto', random_state=42, reg=0.001, tol=1e-06)

    X_reduced = lle.transform(X)

    plt.clf()
    plt.title('LLE results n_neighbors={}'.format(neighbors[i]), fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=10)
    plt.ylabel("$z_2$", fontsize=10)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.savefig('1_4_LLE_results{}'.format(neighbors[i]))


# t-SNE

from sklearn.manifold import TSNE

perplexity = [2, 3, 5, 10, 20]
for i in range(len(perplexity)):
    tsne = TSNE(n_components=2, 
                perplexity = perplexity[i],
                random_state=42)
    X_reduced_tsne = tsne.fit_transform(X)

    plt.clf()
    plt.title('t-SNE_perplexity{}'.format(perplexity[i]), fontsize=14)
    plt.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=y, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=10)
    plt.ylabel("$z_2$", fontsize=10, rotation=0)
    plt.savefig('1_4_t-SNE{}'.format(perplexity[i]))

