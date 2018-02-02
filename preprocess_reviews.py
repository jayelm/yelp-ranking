"""
Preprocess dataset/review.json
"""

from collections import defaultdict, namedtuple
import json
from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np

REVIEW_FILE = 'dataset/review.json'

USERS_FILE = 'dataset_processed/users.pkl'
BUSINESSES_FILE = 'dataset_processed/businesses.pkl'
MATCHES_FILE = 'dataset_processed/matches.pkl'

Review = namedtuple('Review', ['user_id', 'business_id', 'stars'])


def review_from_json_str(json_str):
    review_json = json.loads(json_str)
    return Review(review_json['user_id'],
                  review_json['business_id'],
                  review_json['stars'])


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Preprocess reviews',
        formatter_class=ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    all_reviews = defaultdict(list)
    all_users = set()
    business_ratings = defaultdict(list)

    n_lines = file_len(REVIEW_FILE)
    print("{} reviews".format(n_lines))

    # Load reviews
    with open(REVIEW_FILE, 'r') as rf:
        for line in tqdm(rf, total=n_lines, desc='Loading reviews'):
            review = review_from_json_str(line)
            all_reviews[review.user_id].append(review)
            all_users.add(review.user_id)
            business_ratings[review.business_id].append(review.stars)

    print("{} users".format(len(all_users)))
    print("{} businesses".format(len(business_ratings.keys())))

    avg_business_records = [(b, sum(rs) / len(rs), len(rs))
                            for b, rs in business_ratings.items()]
    business_df = pd.DataFrame(
        avg_business_records,
        columns=['business_id', 'avg_rating', 'n_reviews'],
    )
    # Smaller dtypes to save space
    assert business_df.n_reviews.min() > 0
    assert business_df.n_reviews.max() < np.iinfo(np.uint16).max

    business_df.n_reviews = business_df.n_reviews.astype(np.uint16)
    business_df.avg_rating = business_df.avg_rating.astype(np.float32)
    business_df.business_id = business_df.business_id.astype(np.character)

    business_df.to_pickle(BUSINESSES_FILE)
    businesses_to_ids = dict(zip(business_df.business_id, business_df.index))

    # Convert users and businesses into integer ids
    users_df = pd.DataFrame(list(all_users),
                            columns=['user_id'])
    users_df.user_id = users_df.user_id.astype(np.character)
    users_df.to_pickle(USERS_FILE)
    users_to_ids = dict(zip(users_df.user_id, users_df.index))

    review_matches = []

    for user, user_reviews in tqdm(all_reviews.items(),
                                   total=len(all_reviews.keys()),
                                   desc="Creating matches"):
        uid = users_to_ids[user]
        review_tuples = [(businesses_to_ids[r.business_id], r.stars)
                         for r in user_reviews]
        for r1, r2 in itertools.combinations(review_tuples, 2):
            if r1[1] > r2[1]:
                win = 1
            elif r1[1] < r2[1]:
                win = -1
            else:
                win = 0  # Draw
            review_matches.append((uid, r1[0], r2[0], win))

    print("Creating dataframe")
    matches_df = pd.DataFrame(review_matches,
                              columns=['user', 'b1', 'b2', 'win'])
    # Better dtypes
    assert matches_df.user.min() > 0
    assert matches_df.user.max() < np.iinfo(np.uint32).max
    assert matches_df.b1.min() > 0
    assert matches_df.b1.max() < np.iinfo(np.uint32).max
    assert matches_df.b2.min() > 0
    assert matches_df.b2.max() < np.iinfo(np.uint32).max

    matches_df.user = matches_df.user.astype(np.uint32)
    matches_df.b1 = matches_df.b1.astype(np.uint32)
    matches_df.b2 = matches_df.b2.astype(np.uint32)
    matches_df.win = matches_df.win.astype(np.int8)

    print("Saving to file")
    matches_df.to_pickle(MATCHES_FILE)
