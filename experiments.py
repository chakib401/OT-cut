import scipy
from scipy.io import loadmat

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from ot_cut import ot_cut
from utils import CA, CF1
from time import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import sparse

for dname in [
    'India_database',
    'EmailEU_database',
    "dblp",
    "acm",
    'mnist-lsr',
    'kmnist-lsr',
    'fmnist-lsr',
    'mnist-ensc',
    'fmnist-ensc',
    'kmnist-ensc',
    'mnist-lrr',
    'kmnist-lrr',
    'fmnist-lrr',

]:
    data = loadmat(f'graphs/{dname}')
    A, labels = data['adj'], data['y'].reshape(-1)
    n_clusters = len(np.unique(labels))

    print(dname)
    use_ncut = False
    n_iter = 30
    runs = 5

    # main

    losses = []
    metrics = {'OT' + 'time': [], 'OT' + 'acc': [], 'OT' + 'ami': [],
               'OT' + 'ari': [], 'OT' + 'f1': [], 'OT' + 'KL': []
               }

    for _ in range(runs):
        if use_ncut:
            # ot-ncut
            d = A.sum(1)
            if sparse.issparse(A):
                d = np.array(d)
            d = d.reshape(-1)
            source = d / d.sum()
            target = np.ones(n_clusters)
            target = target / target.sum()
        else:
            # ot-rcut
            source = np.ones(A.shape[0])
            _, target = np.unique(labels, return_counts=True)

        t0 = time()
        ot_predictions, run_losses = ot_cut(A, n_clusters, source, target,
                                            n_iter=n_iter)
        ot_time = time() - t0

        metrics['OT' + 'time'].append(ot_time)
        metrics['OT' + 'acc'].append(CA(labels, ot_predictions))
        metrics['OT' + 'ami'].append(ami(labels, ot_predictions))
        metrics['OT' + 'ari'].append(ari(labels, ot_predictions))
        metrics['OT' + 'f1'].append(CF1(labels, ot_predictions))
        counts = np.unique(ot_predictions, return_counts=True)[1]
        metrics['OT' + 'KL'].append(scipy.special.kl_div(sorted(target) / np.sum(target),
                                                         sorted(counts) / np.sum(counts)))
        losses.append(run_losses)

    results = {
        'mean': {k: (np.mean(v)).round(4) for k, v in metrics.items()},
        'std': {k: (np.std(v)).round(4) for k, v in metrics.items()}
    }

    means = results['mean']
    std = results['std']

    starting_letter = 'n' if use_ncut else 'r'
    print(
        f"OT-{starting_letter}cut & {means['OT''ari']}±{std['OT''ari']}",
        sep=',')
    print(f"{means['OT''time']:.2f}±{std['OT''time']:.2f}")
    print(f"KL {means['OT''KL']:.4f}±{std['OT''KL']:.4f}")
