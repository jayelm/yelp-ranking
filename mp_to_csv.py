"""
Join message passing samples with businesses/matches stats, and save to file
"""

from run_trueskill_mp import (
    BUSINESSES_FILE, MATCHES_FILE, convert_matches_format
)
import pandas as pd
import numpy as np
import scipy.stats
import os

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='mp_to_csv',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('mp_file', type=str,
                        help='MP samples file to load')
    parser.add_argument('--bfile', default=BUSINESSES_FILE,
                        help='Businesses file')
    parser.add_argument('--mfile', default=MATCHES_FILE,
                        help='Matches file')
    parser.add_argument('--gzip', action='store_true',
                        help='Gzip the CSV file')

    args = parser.parse_args()

    businesses = pd.read_pickle(args.bfile)
    matches = pd.read_pickle(args.mfile)
    # Count draws
    draws = matches[matches.win == 0]
    draws_counter = np.bincount(matches.b1) + np.bincount(matches.b2)
    # Drop draws and count wins
    matches = matches[matches.win != 0]
    matches = convert_matches_format(matches)
    wins_counter = np.bincount(matches.b1)
    losses_counter = np.bincount(matches.b2)
    matches_counter = wins_counter + losses_counter + draws_counter

    # Add matches, wins, losses
    businesses['matches'] = matches_counter
    businesses['wins'] = wins_counter
    businesses['losses'] = losses_counter
    businesses['draws'] = draws_counter

    businesses = businesses.rename(columns={'avg_rating': 'star_rating'})

    mp_samples = np.load(args.mp_file)
    ratings, variances = mp_samples[-1, 0, :], mp_samples[-1, 1, :]
    n_b = len(ratings)
    print("{} ratings ({:.3f}, {:.3f})".format(n_b, ratings.min(), ratings.max()))

    businesses['ts_rating'] = ratings
    businesses['ts_variance'] = variances
    businesses['ranking'] = scipy.stats.rankdata(-ratings)

    businesses.business_id = businesses.business_id.astype(str)

    csv_name = 'results/businesses_{}'.format(os.path.basename(args.mp_file))
    csv_name = csv_name.replace('.npy', '.feather')
    businesses.reset_index().to_feather(csv_name)
