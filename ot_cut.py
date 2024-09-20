import numpy as np
import ot
from scipy import sparse

def loss_fn(plan, laplacian_matrix, alpha):
    """Calculate the loss function."""
    return np.sum(np.multiply(laplacian_matrix @ plan, plan)) - .5 / alpha * np.linalg.norm(plan) ** 2

def initialize_plan(num_samples, num_clusters, initial_plan):
    """Initialize the transport plan."""
    plan = np.zeros((num_samples, num_clusters))
    plan[np.arange(num_samples), initial_plan] = 1
    return plan


def ot_cut(adjacency_matrix, num_clusters, source_distribution, target_distribution, n_iter=20, init=None, n_init=1):
    """
    Optimal Transport Cut (OT-cut) algorithm.

    Parameters:
    -----------
    adjacency_matrix : array-like or sparse matrix, shape (n_samples, n_samples)
        The adjacency matrix of the graph.
    num_clusters : int
        The number of clusters.
    source_distribution : array-like, shape (n_samples,)
        The source distribution for optimal transport.
    target_distribution : array-like, shape (n_samples,)
        The target distribution for optimal transport.
    n_iterations : int, default=20
        The number of iterations for the algorithm.
    initial_plan : array-like, shape (n_samples,), optional
        Initial transport plan. If None, a random initialization is used.
    n_init : int, default=1
        The number of different initializations to try (if random).

    Returns:
    --------
    best_labels : array-like, shape (n_samples,)
        The best cluster labels found.
    best_losses : list of float
        The loss values for the best initialization.
    """
    num_samples = len(source_distribution)
    if init is None:
        init = np.random.randint(low=0, high=num_clusters, size=num_samples)
    else:
        n_init = 1

    best_labels = 0
    best_losses = [np.inf]

    for _ in range(n_init):
        plan = initialize_plan(num_samples, num_clusters, init)
        plan = ot.emd(source_distribution, target_distribution, -plan)

        laplacian_matrix = sparse.eye(adjacency_matrix.shape[0]) - adjacency_matrix
        alpha = 1 / 2

        matrix = 2 * alpha * laplacian_matrix - sparse.eye(adjacency_matrix.shape[0])

        losses = []
        tk_1, tk = 0, 1
        xk1 = xk_1 = xk = zk = plan

        for t in range(n_iter):
            yk = xk + tk_1 / tk * (zk - xk) + (tk_1 - 1) / tk * (xk - xk_1)
            zk1 = ot.emd(source_distribution, target_distribution, matrix @ yk)
            vk1 = ot.emd(source_distribution, target_distribution, matrix @ xk)
            tk1 = (np.sqrt(4 * tk ** 2 + 1) + 1) / 2
            z_loss = loss_fn(zk1, laplacian_matrix, alpha)
            v_loss = loss_fn(vk1, laplacian_matrix, alpha)
            xk1 = zk1 if v_loss > z_loss else vk1

            xk_1, xk, zk, tk_1, tk = xk, xk1, zk1, tk, tk1
            losses.append(min(z_loss, v_loss))

        labels = xk1.argmax(-1)

        # Update the best labels and losses based on the last loss value
        if losses[-1] < best_losses[-1]:
            best_losses, best_labels = losses, labels

    return best_labels