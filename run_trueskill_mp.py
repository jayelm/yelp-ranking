"""Message passing trueskill
"""

import pandas as pd
import scipy.stats
import numpy as np
from preprocess_reviews import BUSINESSES_FILE, MATCHES_FILE
from tqdm import tqdm, trange


def psi(x):
    return scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)


def Lambda(x):
    return psi(x) * (psi(x) + x)


def gaussian_ep(M, n_players):
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

    parser.add_argument('--num_samples', default=100,
                        help='Number of message passing samples')

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
    gaussian_ep_with_progress = zip(
        trange(args.num_samples, desc='MP'),
        gaussian_ep(matches, n_businesses))
    for it, mean_and_stdev in gaussian_ep_with_progress:
        mp_samples[it, :] = np.array(mean_and_stdev)
