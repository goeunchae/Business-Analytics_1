import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

# ISOMAP
# without sklearn

from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.utils.graph import graph_shortest_path


def make_adjacency(data, dist_func="euclidean", eps=1):


   n, m = data.shape
   dist = cdist(data.T, data.T, metric=dist_func)
   adj =  np.zeros((m, m)) + np.inf
   bln = dist < eps
   adj[bln] = dist[bln]
   short = graph_shortest_path(adj)

   return short


def isomap(d, dim=2):

    n, m = d.shape
    h = np.eye(m) - (1/m)*np.ones((m, m))
    d = d**2
    c = -1/(2*m) * h.dot(d).dot(h)
    evals, evecs = linalg.eig(c)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals = evals[:dim]
    evecs = evecs[:, :dim]
    z = evecs.dot(np.diag(evals**(-1/2)))

    return z.real


def plot_graph(components, x, my_title="ISOMAP"):

    n, m = x.shape
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111)
    ax.set_title(my_title)
    ax.set_xlabel('Component: 1')
    ax.set_ylabel('Component: 2')

    # Show 2D components plot
    ax.scatter(components[:, 0], components[:, 1], marker='.',alpha=0.7)

    plt.savefig('1_4_isomap_results')
    return None

if __name__ == "__main__":
    df = pd.read_csv('./data/data_wine.csv')
    X = df.drop(['quality'],axis=1)
    y = df['quality'].tolist()
    D = make_adjacency(X, eps=1e+6, dist_func="euclidean")
    #D = make_adjacency(X, eps=386, dist_func="cityblock")
    z = isomap(D)
    plot_graph(z, x=X, my_title="isomap result")

# # with sklearn

from sklearn import manifold

iso = manifold.Isomap(n_neighbors=11, n_components=2)
df = pd.read_csv('./data/data_wine.csv')
X = df.drop(['quality'], axis=1)
y = df['quality'].tolist()
iso.fit(X)
manifold_2Da = iso.transform(X)
manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])


fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.set_title('ISOMAP with sklearn')
ax.set_xlabel('Component: 1')
ax.set_ylabel('Component: 2')

# Show 2D components plot
ax.scatter(manifold_2D['Component 1'], manifold_2D['Component 2'], marker='.',alpha=0.7)
plt.savefig('1_4_isomap_results_with sklearn')

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

