import numpy as np
import numpy.testing as npt

import optlearner


def test_update():
    """Test the fast upate code against a for loop."""
    learner = optlearner.VolatilityLearner()

    for reward in [0, 1]:
        slow_pIk = slow_update(learner, reward)
        learner._update(reward)
        yield npt.assert_array_equal, slow_pIk, learner.pIk
        learner.reset()


def slow_update(learner, reward):
    """This is a more or less literal translation of the original code."""
    pIk = learner.pIk.copy()

    k_grid = learner.k_grid
    I_grid = learner.I_grid
    p_grid = learner.p_grid

    Ip1gIk = learner._I_trans
    pp1gpIp1 = learner._p_trans

    for k in xrange(k_grid.size):

        # 1) Multiply pIk by Ip1gIk and integrate out I. This will give pIp1k
        pIp1k = np.zeros((p_grid.size, I_grid.size))
        for Ip1 in xrange(I_grid.size):
            for p in xrange(p_grid.size):
                pIp1k[p, Ip1] = np.sum(Ip1gIk[Ip1, :, k] * pIk[p, :, k])

        # 2) Multiply pIp1k by pp1gpIp1 and integrate out p.
        pp1Ip1k = np.zeros((p_grid.size, I_grid.size))
        for Ip1 in xrange(I_grid.size):
            for pp1 in xrange(p_grid.size):
                pp1Ip1k[pp1, Ip1] = np.sum(pIp1k[:, Ip1] *
                                           pp1gpIp1[pp1, :, Ip1].T)

        # 3) Place pp1Ip1k into pIk (belief that is carried to the next trial)
        pIk[:, :, k] = pp1Ip1k

    if reward:
        for k in xrange(k_grid.size):
            for p in xrange(p_grid.size):
                pIk[p, :, k] *= p_grid[p]
    else:
        for k in xrange(k_grid.size):
            for p in xrange(p_grid.size):
                pIk[p, :, k] *= 1 - p_grid[p]

    # Normalization
    pIk /= pIk.sum()

    return pIk
