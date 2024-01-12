import numpy as np
import scipy as sp
from scipy.io import savemat
from sklearn.utils.extmath import randomized_svd
from cluster.selfrepresentation import ElasticNetSubspaceClustering
from utils import get_dataset

for method in [
    'lsr',
    'ensc',
    'lrr'
]:

    for dname in [
        'mnist',
        'kmnist',
        'fmnist',
    ]:
        x, labels = get_dataset(dname)
        n_clusters = len(np.unique(labels))
        if method == 'lsr':
            u, e, _ = randomized_svd(x, n_components=min(x.shape), n_iter=4)
            A = (u * (e ** 2 / (e ** 2 + 100))) @ u.T
        if method == 'lrr':
            u, e, _ = randomized_svd(x, n_components=20, n_iter=4)
            A = u @ u.T
        if method == 'ensc':
            model = ElasticNetSubspaceClustering(n_clusters=n_clusters,
                                                 algorithm='lasso_lars', gamma=50).fit(x)
            A = model.representation_matrix_.A

        A = np.abs(A)
        A = (A + A.T) / 2
        s = A.sum(0).reshape(-1) ** -.5
        s[~np.isfinite(s)] = 0
        A = sp.sparse.diags(s) @ A @ sp.sparse.diags(s)

        savemat(f'graphs/{dname}-{method}.mat', {'adj': A, 'y': labels})
