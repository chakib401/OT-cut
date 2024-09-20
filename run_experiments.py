import scipy
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score as ari
from ot_cut import ot_cut
from time import time
import warnings
import numpy as np
from scipy import sparse

warnings.filterwarnings('ignore')


def compute_source_target(adjacency_matrix, labels, n_clusters, use_ncut):
    if use_ncut:
        d = adjacency_matrix.sum(1)
        if sparse.issparse(adjacency_matrix):
            d = np.array(d).reshape(-1)
        source = d / d.sum()
        target = np.ones(n_clusters) / n_clusters
    else:
        source = np.ones(adjacency_matrix.shape[0])
        _, target = np.unique(labels, return_counts=True)
    return source, target

def run_experiment(adjacency_matrix, labels, n_clusters, source, target, n_iter, runs):
    metrics = {'time': [], 'ari': [], 'KL': []}

    for _ in range(runs):
        t0 = time()
        ot_predictions = ot_cut(adjacency_matrix, n_clusters, source, target, n_iter=n_iter)
        ot_time = time() - t0

        metrics['time'].append(ot_time)
        metrics['ari'].append(ari(labels, ot_predictions))
        counts = np.unique(ot_predictions, return_counts=True)[1]
        metrics['KL'].append(scipy.special.kl_div(sorted(target) / np.sum(target), sorted(counts) / np.sum(counts)))

    return metrics


def main():
    for dataset_name in [
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
        # Print the results in yaml style
        print(f"{dataset_name}:")
        for use_ncut in [True, False]:
            # Load the dataset
            data = loadmat(f'graphs/{dataset_name}')
            adjacency_matrix, labels = data['adj'], data['y'].reshape(-1)
            n_clusters = len(np.unique(labels))

            # Create the source and target distributions
            source, target = compute_source_target(adjacency_matrix, labels, n_clusters, use_ncut)

            # Run the experiment
            metrics = run_experiment(adjacency_matrix, labels, n_clusters, source, target, n_iter=30, runs=5)

            results = {
                'mean': {k: (np.mean(v)).round(4) for k, v in metrics.items()},
                'std': {k: (np.std(v)).round(4) for k, v in metrics.items()}
            }


            # Continue printing the results in yaml style

            means = results['mean']
            std = results['std']

            print(f"  OT-{'n' if use_ncut else 'r'}cut:")
            print(f"    ARI: {means['ari']:.4f} ± {std['ari']:.4f}")
            print(f"    Time: {means['time']:.2f} ± {std['time']:.2f}")
            print(f"    KL-divergence: {means['KL']:.4f} ± {std['KL']:.4f}")

if __name__ == "__main__":
    main()
