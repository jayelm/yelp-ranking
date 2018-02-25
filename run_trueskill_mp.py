"""Message passing trueskill
"""

import pandas as pd
from scipy.stats import norm
import numpy as np
from preprocess_reviews import BUSINESSES_FILE, MATCHES_FILE
from tqdm import tqdm, trange


# Aliases
pdf = norm.pdf
cdf = norm.cdf
ppf = norm.ppf


def psi(x, e):
    return pdf(x - e) / cdf(x - e)


def psidraw(x, e):
    return (
        (pdf(-e - x) - pdf(e - x)) /
        (cdf(e - x) - cdf(-e - x))
    )


def Lambda(x, e):
    return psi(x, e) * (psi(x, e) + (x - e))


def Lambdadraw(x, e):
    return (
        psidraw(x, e) ** 2 +
        (
            ((e - x) * pdf(e - x) + (e + x) * pdf(e + x)) /
            (cdf(e - x) - cdf(-e - x))
        )
    )


def draw_p_to_eps(p):
    """
    Draw probability to epsilon draw value
    """
    return ppf((p + 1.0) / 2)


def gaussian_ep(M, n_players, D, fast_m_acc=False,
                draw_p=0.1):
    assert D.shape[0] == M.shape[0]
    N = len(M)
    it = 0
    epsilon = draw_p_to_eps(draw_p)

    mu_s, p_s = np.empty(n_players, dtype=np.float32), np.empty(n_players, dtype=np.float32)
    mu_gs, p_gs = np.zeros((N, 2), dtype=np.float32), np.zeros((N, 2), dtype=np.float32)
    mu_sg, p_sg = np.empty((N, 2), dtype=np.float32), np.empty((N, 2), dtype=np.float32)

    while True:
        # 1. Compute marginal skills
        # Let skills be N(mu_s, 1/p_s)
        p_s = np.ones(n_players, dtype=np.float32) * 1 / 0.5
        mu_s = np.zeros(n_players, dtype=np.float32)

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
        eps = epsilon * (1 / sigma_gt)
        # Depending on draws, compute different scores
        mu_update = np.where(
            D,
            psidraw(mu_gt / sigma_gt, eps),
            psi(mu_gt / sigma_gt, eps)
        )
        mu_t = mu_gt + sigma_gt * mu_update

        p_update = np.where(
            D,
            Lambdadraw(mu_gt / sigma_gt, eps),
            Lambda(mu_gt / sigma_gt, eps)
        )
        p_t = 1 / v_gt / (1 - p_update)

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
    # Order doesn't matter for draws
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

    parser.add_argument('--bfile', default=BUSINESSES_FILE,
                        help='Business file to load')
    parser.add_argument('--mfile', default=MATCHES_FILE,
                        help='Matches file to load')

    parser.add_argument('--num_samples', default=10, type=int,
                        help='Number of message passing samples')

    parser.add_argument(
        '--save',
        default='results/mp_{num_samples}.npy',
        type=str,
        help='Where to save')

    parser.add_argument(
        '--fast_m_acc',
        action='store_true',
        help='Use cython-optimized accumulator')

    args = parser.parse_args()

    if args.fast_m_acc:
        import pyximport
        pyximport.install(
            setup_args={'include_dirs': [np.get_include()]})
        import fast_m_acc as fma

    print("Loading matches")
    matches = pd.read_pickle(args.mfile)
    # Convert such that b1 is the winning column and b2 is the losing column
    businesses = pd.read_pickle(args.bfile)
    n_businesses = businesses.shape[0]

    # Get draws as boolean array
    draws = matches.win == 0

    matches = convert_matches_format(matches).as_matrix()

    print("Running message passing")
    mp_samples = np.zeros((args.num_samples, 2, n_businesses))

    gep = gaussian_ep(matches, n_businesses, draws,
                      fast_m_acc=args.fast_m_acc,
                      # Use empirical draw probability
                      draw_p=draws.mean())
    gep_with_progress = zip(
        trange(args.num_samples, desc='MP'), gep)
    for it, mean_and_stdev in gep_with_progress:
        mp_samples[it, :] = np.array(mean_and_stdev)

    np.save(args.save.format(**vars(args)), mp_samples)
