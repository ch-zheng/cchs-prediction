from sklearn.cluster import KMeans
import numpy as np

X = np.load('data/pruned/samples.npy')
y = np.load('data/pruned/labels.npy')
model = KMeans(3)
labels = model.fit_predict(X)

a = 0 # n_cluster 0
b = 0 # n_cluster 1
c = 0 # cluster 0 occ
d = 0 # cluster 1 occ
e = 0
f = 0
for i in range(len(labels)):
    if labels[i] == 0:
        a += 1
        if y[i] == 1:
            c += 1
    elif labels[i] == 1:
        b += 1
        if y[i] == 1:
            d += 1
    else:
        e += 1
        f += 1
print('A', c / a)
print('B', d / b)
print('C', e / f)
