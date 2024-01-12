import numpy as np
import ot
from scipy import sparse
from tqdm import tqdm


def loss_fn(p, L, alpha):
    return np.sum(np.multiply(L @ p, p)) - 1 / (2 * alpha) * np.linalg.norm(p) ** 2


def ot_cut(W,
           k,
           pi_s,
           pi_t,
           n_iter=20,
           init=None,
           n_init=1,
           ):
    n = len(pi_s)

    if init is None:
        init = np.random.randint(low=0, high=k, size=n)

    b_y = 0
    b_losses = [np.inf]

    for _ in range(n_init):
        plan = np.zeros((n, k))
        plan[np.arange(n), init] = 1
        plan = ot.emd(pi_s, pi_t, -plan)

        L = sparse.eye(W.shape[0]) - W
        l = 2
        alpha = 1 / l

        M = 2 * alpha * L - sparse.eye(W.shape[0])

        losses = []
        tk_1 = 0
        tk = 1
        xk1 = xk_1 = xk = zk = plan

        for t in tqdm(range(n_iter)):
            yk = xk + tk_1 / tk * (zk - xk) + (tk_1 - 1) / tk * (xk - xk_1)
            zk1 = ot.emd(pi_s, pi_t, M @ yk)
            vk1 = ot.emd(pi_s, pi_t, M @ xk)
            tk1 = (np.sqrt(4 * tk ** 2 + 1) + 1) / 2
            z_loss = loss_fn(zk1, L, alpha)
            v_loss = loss_fn(vk1, L, alpha)
            xk1 = zk1 if v_loss > z_loss else vk1

            xk_1 = xk
            xk = xk1
            zk = zk1
            tk_1 = tk
            tk = tk1

            losses.append(min((z_loss, v_loss)))

        y = xk1.argmax(-1)

        if losses[-1] < b_losses[-1]:
            b_losses = losses
            b_y = y

    return b_y, b_losses
