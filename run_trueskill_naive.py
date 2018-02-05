"""
Run the trueskill algorithm
"""

import pandas as pd
import trueskill as ts
from collections import defaultdict
from tqdm import tqdm

MATCHES_DF = 'dataset_processed/matches.pkl'

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='run trueskill',
        formatter_class=ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    matches = pd.read_pickle(MATCHES_DF)

    ts.setup(backend='scipy')

    ratings = defaultdict(ts.Rating)

    for match in tqdm(matches.itertuples(), total=matches.shape[0],
                      desc='Computing matches'):
        b1, b2 = match.b1, match.b2
        if match.win == 1:
            # b1 wins
            ratings[b1], ratings[b2] = ts.rate_1vs1(ratings[b1], ratings[b2])
        elif match.win == -1:
            # b2 wins
            ratings[b2], ratings[b1] = ts.rate_1vs1(ratings[b2], ratings[b1])
        else:
            ratings[b1], ratings[b2] = ts.rate_1vs1(ratings[b1], ratings[b2],
                                                    drawn=True)
