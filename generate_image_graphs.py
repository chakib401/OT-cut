# [1] Lu, Can-Yi, et al. "Robust and efficient subspace segmentation via least squares regression." Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VII 12. Springer Berlin Heidelberg, 2012.
# [2] Liu, Guangcan, Zhouchen Lin, and Yong Yu. "Robust subspace segmentation by low-rank representation." Proceedings of the 27th international conference on machine learning (ICML-10). 2010.
# [3] You, Chong, et al. "Oracle based active set algorithm for scalable elastic net subspace clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

import numpy as np
import scipy as sp
from scipy.io import savemat
from sklearn.utils.extmath import randomized_svd
from elastic_net.selfrepresentation import ElasticNetSubspaceClustering
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
            # Generate the affinity matrix using the LSR method [1].
            u, e, _ = randomized_svd(x, n_components=min(x.shape), n_iter=4)
            A = (u * (e ** 2 / (e ** 2 + 100))) @ u.T
        if method == 'lrr':
            # Generate the affinity matrix using the LRR method [2].
            u, e, _ = randomized_svd(x, n_components=20, n_iter=4)
            A = u @ u.T
        if method == 'ensc':
            # Generate the affinity matrix using the ENSC method [3].
            model = ElasticNetSubspaceClustering(n_clusters=n_clusters,
                                                 algorithm='lasso_lars', gamma=50).fit(x)
            A = model.representation_matrix_.A

        A = np.abs(A)
        A = (A + A.T) / 2
        s = A.sum(0).reshape(-1) ** -.5
        s[~np.isfinite(s)] = 0
        A = sp.sparse.diags(s) @ A @ sp.sparse.diags(s)

        savemat(f'graphs/{dname}-{method}.mat', {'adj': A, 'y': labels})
