#coding:utf-8
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import Isomap,locally_linear_embedding,spectral_embedding
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

data=np.loadtxt('swiss-data.txt')
[n,m]=np.shape(data)
X=data[0:n,1:m]
Y=data[0:n,0]

N1=(Y==1)
N2=(Y==2)
N3=(Y==3)

pca=PCA(copy=True, n_components=2, whiten=False)
pcaX=pca.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
ldaX = lda.fit(X, Y).transform(X)

kpca = KernelPCA(kernel="rbf",n_components=2)
kpcaX = kpca.fit_transform(X)

isomap=Isomap(n_neighbors=15,n_components=2)
isomapX=isomap.fit_transform(X)

iieX,err=locally_linear_embedding(X,n_neighbors=12,n_components=2)
print("Done. Reconstruction error: %g" % err)


distX=np.zeros([len(Y),len(Y)])
nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
for i in range(0,len(Y)):
    distX[indices[i,0],indices[i,1:15]]=distances[i,1:15]
print(distX)
leX=spectral_embedding(distX,n_components=2)

plt.figure()
ax=plt.subplot(projection='3d')
ax.scatter(X[N1, 0], X[N1, 1], X[N1, 2], c='b')
ax.scatter(X[N2, 0], X[N2, 1], X[N2, 2], c='g')
ax.scatter(X[N3, 0], X[N3, 1], X[N3, 2], c='r')
plt.figure()
plt.scatter(pcaX[N1,0],pcaX[N1,1], color='b')
plt.scatter(pcaX[N2,0],pcaX[N2,1], color='g')
plt.scatter(pcaX[N3,0],pcaX[N3,1], color='r')
plt.figure()
plt.scatter(ldaX[N1,0],ldaX[N1,1], color='b')
plt.scatter(ldaX[N2,0],ldaX[N2,1], color='g')
plt.scatter(ldaX[N3,0],ldaX[N3,1], color='r')
plt.figure()
plt.scatter(kpcaX[N1,0],kpcaX[N1,1], color='b')
plt.scatter(kpcaX[N2,0],kpcaX[N2,1], color='g')
plt.scatter(kpcaX[N3,0],kpcaX[N3,1], color='r')
plt.figure()
plt.scatter(isomapX[N1,0],isomapX[N1,1], color='b')
plt.scatter(isomapX[N2,0],isomapX[N2,1], color='g')
plt.scatter(isomapX[N3,0],isomapX[N3,1], color='r')
plt.figure()
plt.scatter(iieX[N1,0],iieX[N1,1], color='b')
plt.scatter(iieX[N2,0],iieX[N2,1], color='g')
plt.scatter(iieX[N3,0],iieX[N3,1], color='r')
plt.figure()
plt.scatter(leX[N1,0],leX[N1,1], color='b')
plt.scatter(leX[N2,0],leX[N2,1], color='g')
plt.scatter(leX[N3,0],leX[N3,1], color='r')
plt.show()