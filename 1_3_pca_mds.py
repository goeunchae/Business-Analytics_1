import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings(action='ignore')


df = pd.read_csv('./data/data_wine.csv')
X = df.drop(['quality'],axis=1)
y = df['quality'].tolist()

## PCA
# without sklearn 
# Standardize the data
X = (X - X.mean()) / X.std(ddof=0)
# Calculating the correlation matrix of the data
X_corr = (1 / 150) * X.T.dot(X)

eig_values, eig_vectors = np.linalg.eig(X_corr)

# plotting the variance explained by each PC 
explained_variance=(eig_values / np.sum(eig_values))*100
plt.figure(figsize=(8,4))
plt.bar(range(11), explained_variance, alpha=0.6)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Dimensions')
#plt.show()

# calculating our new axis
pc1 = X.dot(eig_vectors[:,0])
pc2 = X.dot(eig_vectors[:,1])

def plot_scatter(pc1, pc2):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    unique = list(set(y))
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    
    for i, spec in enumerate(y):
        plt.scatter(pc1[i], pc2[i], label = spec, s = 20, c=colors[unique.index(spec)])
        ax.annotate(str(i+1), (pc1[i],pc2[i]))
    
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 15}, loc=4)
    
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.axvline(x=0, color="grey", linestyle="--")
    
    plt.clf()
    plt.grid()
    plt.axis([-4, 4, -3, 3])
    plt.show()
    
#plot_scatter(pc1, pc2)

# with sklearn 
from sklearn.decomposition import PCA

pca = PCA()
result = pca.fit_transform(X)
# Remember what we said about the sign of eigen vectors that might change ?
pc1 = - result[:,0]
pc2 = - result[:,1]
#plot_scatter(pc1, pc2)


## MDS
# without sklearn 
B = X.dot(X.T)

eig_values, eig_vectors = np.linalg.eig(B)

mds_axis = eig_vectors[:,0]
proj_XonMDS = np.linalg.norm(mds_axis.dot(X)/np.linalg.norm(mds_axis))*mds_axis/np.linalg.norm(mds_axis)
#print('For q = 1 Project of X vector on Y ∈ R(3*1) is:\n', proj_XonMDS)


mds_axis = eig_vectors[:,1]
proi_XonMDS = np.linalg.norm(mds_axis.dot(X)/np.linalg.norm(mds_axis))*mds_axis/np.linalg.norm(mds_axis)
#print('For q = 1 Project of X vector on Y ∈ R(3*1) is:\n', proi_XonMDS)
plt.clf()
plt.scatter(proi_XonMDS,proj_XonMDS)
plt.xlabel('Coordinate 1')
plt.ylabel('Coordinate 2')
plt.savefig('1_3_MDS')

# # with sklearn
from sklearn.manifold import MDS

mds = MDS(random_state=0)
scaled_df = mds.fit_transform(X)
#create scatterplot
plt.clf()
plt.scatter(scaled_df[:,0], scaled_df[:,1])

#add axis labels
plt.xlabel('Coordinate 1')
plt.ylabel('Coordinate 2')

#add lables to each point
for i, txt in enumerate(df.index):
    plt.annotate(txt, (scaled_df[:,0][i]+.3, scaled_df[:,1][i]))

plt.show()
