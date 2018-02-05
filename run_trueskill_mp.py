"""Message passing trueskill
"""

import pandas as pd
import scipy.stats
import numpy as np
from preprocess_reviews import BUSINESSES_FILE, MATCHES_FILE
from tqdm import tqdm, trange

import traceback
import ctypes
import multiprocessing as mp
from collections import namedtuple

import pyximport
pyximport.install(
    setup_args={'include_dirs': [np.get_include()]})
import fast_m_acc


MPArgs = namedtuple('MPArgs', ['mu_s', 'p_s',
                               'winner', 'loser', 'j',
                               'p_gs', 'mu_gs'])


def psi(x):
    return scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)


def Lambda(x):
    return psi(x) * (psi(x) + x)


def shared_array(N):
    """
    Form a shared memory numpy array.

    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """
    shared_array_base = mp.Array(ctypes.c_double, N)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    return shared_array


def shared_array_2d(shape):
    """
    Form a shared memory numpy array.

    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """
    shared_array_base = mp.Array(ctypes.c_double, shape[0] * shape[1])
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array


def incr_p_and_mu(mpargs):
    """
    Function that operates on shared memory.
    """
    try:
        # Modify ps
        p_gs_j_0 = mpargs.p_gs[mpargs.j, 0]
        p_gs_j_1 = mpargs.p_gs[mpargs.j, 1]
        lock_p.acquire()
        mpargs.p_s[mpargs.winner] += p_gs_j_0
        mpargs.p_s[mpargs.loser] += p_gs_j_1
        lock_p.release()

        # Modify mus
        mu_gs_j_0 = mpargs.mu_gs[mpargs.j, 0]
        mu_gs_j_1 = mpargs.mu_gs[mpargs.j, 1]
        lock_mu.acquire()
        mpargs.mu_s[mpargs.winner] += mu_gs_j_0 * p_gs_j_0
        mpargs.mu_s[mpargs.loser] += mu_gs_j_1 * p_gs_j_1
        lock_mu.release()
    except Exception as e:
        print("Caught exception")
        traceback.print_exc()
        print()
        raise e


def init_locks(lp, lmu):
    global lock_p, lock_mu
    lock_p = lp
    lock_mu = lmu


def gaussian_ep_mp(M, n_players, n_cpu=2):
    print("Warning: this doesn't work")
    N = len(M)
    it = 0

    # Shared arrays for multiprocessing
    mu_s = shared_array(n_players)
    p_s = shared_array(n_players)

    # Multiprocessing setup
    mu_sg, p_sg = np.empty((N, 2)), np.empty((N, 2))

    p_gs = shared_array_2d((N, 2))
    mu_gs = shared_array_2d((N, 2))

    while True:
        # 1. Compute marginal skills
        # Let skills be N(mu_s, 1/p_s)
        p_s[:] = np.ones(n_players) * 1 / 0.5
        mu_s[:] = np.zeros(n_players)
        mpargs_generator = (
            MPArgs(mu_s, p_s,
                   winner, loser, j,
                   p_gs, mu_gs)
            for j, (winner, loser)
            in enumerate(M))

        lock_p = mp.Lock()
        lock_mu = mp.Lock()
        pool = mp.Pool(n_cpu, initializer=init_locks,
                       initargs=(lock_p, lock_mu))
        pool.map(incr_p_and_mu, mpargs_generator, chunksize=1000)
        pool.close()
        pool.join()

        mu_s[:] = mu_s / p_s

        # 2. Compute skill -> game messages
        # winner's skill -> game: N(mu_sg[,0], 1/p_sg[,0])
        # loser's skill -> game: N(mu_sg[,1], 1/p_sg[,1])
        p_sg = p_s[M] - p_gs
        mu_sg = (p_s[M] * mu_s[M] - p_gs * mu_gs) / p_sg

        # 3. Compute game -> performance messages
        v_gt = 1 + np.sum(1 / p_sg, 1)
        sigma_gt = np.sqrt(v_gt)
        mu_gt = mu_sg[:, 0] - mu_sg[:, 1]

        # 4. Approximate the marginal on performance differences
        mu_t = mu_gt + sigma_gt * psi(mu_gt / sigma_gt)
        p_t = 1 / v_gt / (1 - Lambda(mu_gt / sigma_gt))

        # 5. Compute performance -> game messages
        p_tg = p_t - 1 / v_gt
        mu_tg = (mu_t * p_t - mu_gt / v_gt) / p_tg

        # 6. Compute game -> skills messages
        # game -> winner's skill: N(mu_gs[,0], 1/p_gs[,0])
        # game -> loser's skill: N(mu_gs[,1], 1/p_gs[,1])
        p_gs[:, 0] = 1 / (1 + 1 / p_tg + 1 / p_sg[:, 1])  # winners
        p_gs[:, 1] = 1 / (1 + 1 / p_tg + 1 / p_sg[:, 0])  # losers
        mu_gs[:, 0] = mu_sg[:, 1] + mu_tg
        mu_gs[:, 1] = mu_sg[:, 0] - mu_tg

        it += 1

        yield (mu_s, np.sqrt(1 / p_s))


def gaussian_ep(M, n_players, fast_m_acc=False):
    N = len(M)
    it = 0

    mu_s, p_s = np.empty(n_players), np.empty(n_players)
    mu_gs, p_gs = np.zeros((N, 2)), np.zeros((N, 2))
    mu_sg, p_sg = np.empty((N, 2)), np.empty((N, 2))

    while True:
        # 1. Compute marginal skills
        # Let skills be N(mu_s, 1/p_s)
        p_s = np.ones(n_players) * 1 / 0.5
        mu_s = np.zeros(n_players)

        if fast_m_acc:
            fma.fast_m_acc(M, p_s, mu_s, p_gs, mu_gs, N)
        else:
            M_with_progress = tqdm(M, total=N, desc='MP iter {}'.format(it))

            for j, (winner, loser) in enumerate(M_with_progress):
                p_s[winner] += p_gs[j, 0]
                p_s[loser] += p_gs[j, 1]
                mu_s[winner] += mu_gs[j, 0] * p_gs[j, 0]
                mu_s[loser] += mu_gs[j, 1] * p_gs[j, 1]

        mu_s = mu_s / p_s

        # 2. Compute skill -> game messages
        # winner's skill -> game: N(mu_sg[,0], 1/p_sg[,0])
        # loser's skill -> game: N(mu_sg[,1], 1/p_sg[,1])
        p_sg = p_s[M] - p_gs
        mu_sg = (p_s[M] * mu_s[M] - p_gs * mu_gs) / p_sg

        # 3. Compute game -> performance messages
        v_gt = 1 + np.sum(1 / p_sg, 1)
        sigma_gt = np.sqrt(v_gt)
        mu_gt = mu_sg[:, 0] - mu_sg[:, 1]

        # 4. Approximate the marginal on performance differences
        mu_t = mu_gt + sigma_gt * psi(mu_gt / sigma_gt)
        p_t = 1 / v_gt / (1 - Lambda(mu_gt / sigma_gt))

        # 5. Compute performance -> game messages
        p_tg = p_t - 1 / v_gt
        mu_tg = (mu_t * p_t - mu_gt / v_gt) / p_tg

        # 6. Compute game -> skills messages
        # game -> winner's skill: N(mu_gs[,0], 1/p_gs[,0])
        # game -> loser's skill: N(mu_gs[,1], 1/p_gs[,1])
        p_gs[:, 0] = 1 / (1 + 1 / p_tg + 1 / p_sg[:, 1])  # winners
        p_gs[:, 1] = 1 / (1 + 1 / p_tg + 1 / p_sg[:, 0])  # losers
        mu_gs[:, 0] = mu_sg[:, 1] + mu_tg
        mu_gs[:, 1] = mu_sg[:, 0] - mu_tg

        it += 1

        yield (mu_s, np.sqrt(1 / p_s))


def convert_matches_format(matches):
    if np.any(matches.win == 0):
        print("Warning: draws in matches df, don't know what to do")
    matches['b1_temp'] = np.where(matches.win == 1, matches.b1, matches.b2)
    matches['b2_temp'] = np.where(matches.win == 1, matches.b2, matches.b1)
    matches['b1'], matches['b2'] = matches['b1_temp'], matches['b2_temp']
    matches = matches.drop(columns=['b1_temp', 'b2_temp', 'user', 'win'])
    return matches


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Trueskill message passing',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_samples', default=10, type=int,
                        help='Number of message passing samples')

    parser.add_argument('--n_cpu', default=1, type=int,
                        help='Number of CPUS (> 1 enables multiprocessing)')

    parser.add_argument(
        '--save',
        default='results/mp_{draw}draws_{num_samples}.npy',
        type=str,
        help='Where to save')

    parser.add_argument(
        '--draw',
        type=str,
        default='drop',
        choices=['drop', 'double'],
        help='How to handle draws')

    args = parser.parse_args()

    print("Loading matches")
    matches = pd.read_pickle(MATCHES_FILE)
    # Convert such that b1 is the winning column and b2 is the losing column
    businesses = pd.read_pickle(BUSINESSES_FILE)
    n_businesses = businesses.shape[0]

    if args.draw == 'drop':
        print("Dropping draws")
        matches = matches[matches.win != 0]
        matches = convert_matches_format(matches)
    elif args.draw == 'double':
        print("Doubling draw games")
        # Isolate draws, make draws double matches, where both teams win
        non_draws = matches[matches.win != 0]
        non_draws = convert_matches_format(non_draws)
        draws = matches[matches.win == 0].drop(columns=['user', 'win'])
        draws_copy = draws.copy()
        # Reverse draws copy
        draws_copy.b2, draws_copy.b1 = draws_copy.b1, draws_copy.b2

        matches = pd.concat([non_draws, draws, draws_copy])
    else:
        raise NotImplementedError

    assert list(matches.columns) == ['b1', 'b2'], "Matches in wrong format"
    matches = matches.as_matrix()

    print("Running message passing")
    mp_samples = np.zeros((args.num_samples, 2, n_businesses))

    if args.n_cpu > 1:
        gep = gaussian_ep_mp(matches, n_businesses, args.n_cpu)
    else:
        gep = gaussian_ep(matches, n_businesses)
    gep_with_progress = zip(
        trange(args.num_samples, desc='MP'), gep)
    for it, mean_and_stdev in gep_with_progress:
        mp_samples[it, :] = np.array(mean_and_stdev)

    np.save(args.save.format(**vars(args)), mp_samples)
