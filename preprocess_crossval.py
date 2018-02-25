"""
Preprocess dataset/review.json into cross-validation splits
"""

from collections import defaultdict, namedtuple
import json
from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np
import random
import multiprocessing as mp
from sklearn.model_selection import KFold

N_CPU = 6

random.seed(0)
np.random.seed(0)

REVIEW_FILE = 'dataset/review.json'

USERS_FILE = 'dataset_processed/users.pkl.mp'
BUSINESSES_FILE = 'dataset_processed/businesses.pkl'
MATCHES_FILE = 'dataset_processed/matches.pkl'
REVIEWS_FILE = 'dataset_processed/reviews.pkl'

Review = namedtuple(
    'Review',
    ['user_id', 'business_id', 'stars']
)


def to_pickle_and_gz(df, fname, csv=False, gzip=False):
    if gzip and not csv:
        raise ValueError("Cannot set gzip without csv=True")
    df.to_pickle(fname)
    if csv:
        # Get rid of byte strings
        for column in df.columns:
            if df[column].dtype.kind == 'S':
                df[column] = df[column].astype(str)
        gz_fname = fname.replace('.pkl', '.csv.gz')
        df.to_csv(gz_fname,
                  compression='gzip' if gzip else None)


def review_from_json_str(json_str):
    """
    Create a review object from the JSON
    """
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
    parser.add_argument('--csv', action='store_true',
                        help='Save .csv files')
    parser.add_argument('--csv_gzip', action='store_true',
                        help='Gzipped .csv files (takes a while)')

    args = parser.parse_args()

    if args.csv_gzip and not args.csv:
        parser.error("Cannot specify --csv_gzip without --csv")

    n_lines = file_len(REVIEW_FILE)
    print("{} reviews".format(n_lines))

    # Load reviews
    print("Opening reviews")
    with open(REVIEW_FILE, 'r') as rf:
        rf_lines = list(rf)

    # Generate review_objs
    pool = mp.Pool(N_CPU)
    reviews = pool.map(
        review_from_json_str,
        tqdm(rf_lines, total=n_lines, desc='Loading reviews'))
    pool.close()
    pool.join()

    # Initialize counters of reviews
    all_reviews = defaultdict(list)
    all_users = set()
    all_businesses = set()
    business_ratings = defaultdict(list)
    # Collect reviews in counters
    for review in reviews:
        all_reviews[review.user_id].append(review)
        all_users.add(review.user_id)
        all_businesses.add(review.business_id)

    print("{} users".format(len(all_users)))
    print("{} businesses".format(len(all_businesses)))

    # Initialize business_df but leave out ratings/n reviews for now
    business_df = pd.DataFrame(
        sorted(list(all_businesses)),
        columns=['business_id']
    )
    # Make dict mapping businesses to their ids
    businesses_to_ids = dict(zip(business_df.business_id, business_df.index))

    # Since users to ids mapping is constant, we keep this
    # outside of the cross validation loop
    # TODO: We may be interested in how many reviews the user has done
    # though
    users_with_n = sorted(
        [(u, len(rs)) for u, rs in all_reviews.items()],
        key=lambda x: x[0]
    )
    users_df = pd.DataFrame(users_with_n,
                            columns=['user_id', 'n_reviews'])
    assert users_df.n_reviews.min() > 0
    assert users_df.n_reviews.max() < np.iinfo(np.uint16).max
    users_df.n_reviews = users_df.n_reviews.astype(np.uint16)
    users_to_ids = dict(zip(users_df.user_id, users_df.index))
    # Make dict *before* coercing to bytes
    users_df.user_id = users_df.user_id.astype(np.character)
    to_pickle_and_gz(users_df, USERS_FILE,
                     csv=args.csv,
                     gzip=args.csv_gzip)

    # Now do cross-validation on USERS
    all_users = np.array(list(all_users))
    np.random.shuffle(all_users)
    u_splits = KFold(5).split(all_users)
    for k, (u_train_idx, u_test_idx) in enumerate(u_splits):
        u_train = all_users[u_train_idx]
        u_test = all_users[u_test_idx]

        u_train = set(u_train)
        u_test = set(u_test)
        print("Split {}: {} train users, {} test".format(
            k, len(u_train), len(u_test)))
        # Now collect ratings for train users only
        business_ratings = defaultdict(list)
        for review in reviews:
            if review.user_id in u_train:
                business_ratings[review.business_id].append(review.stars)
        avg_business_records = [(b, sum(rs) / len(rs), len(rs))
                                for b, rs in business_ratings.items()]
        split_ratings = pd.DataFrame(
            avg_business_records,
            columns=['business_id', 'avg_rating', 'n_reviews']
        )
        # Merge with original df
        bdf_split = business_df.merge(split_ratings,
                                      how='left',
                                      on='business_id')
        # Leave avg rating NAs alone, since what to do in that case depends on
        # model. (One easy way is just to impute with mean)
        bdf_split.n_reviews = bdf_split.n_reviews.fillna(0.0)
        print("{} reviews in train split (total: {})".format(
            sum(bdf_split.n_reviews),
            sum(len(v) for v in all_reviews.values()))
        )

        # Save bdf_split file
        # Smaller dtypes to save space
        # In this case, n_reviews CAN be 0
        assert not (bdf_split.n_reviews.min() < 0.0)
        assert bdf_split.n_reviews.max() < np.iinfo(np.uint16).max

        bdf_split.n_reviews = bdf_split.n_reviews.astype(np.uint16)
        bdf_split.avg_rating = bdf_split.avg_rating.astype(np.float32)
        bdf_split.business_id = bdf_split.business_id.astype(np.character)

        bfile = BUSINESSES_FILE + '.{}.train'.format(k)
        to_pickle_and_gz(bdf_split, bfile,
                         csv=args.csv,
                         gzip=args.csv_gzip)

        # Now filter reviews by user
        train_reviews = {u: rs for u, rs in all_reviews.items()
                         if u in u_train}
        test_reviews = {u: rs for u, rs in all_reviews.items()
                        if u in u_test}

        for phase, phase_reviews in [('train', train_reviews),
                                     ('test', test_reviews)]:
            # Create reviews only dataframe, with user_id, business_id, and stars
            reviews_records = []
            for user in phase_reviews:
                for review in phase_reviews[user]:
                    reviews_records.append((
                        users_to_ids[review.user_id],
                        businesses_to_ids[review.business_id],
                        review.stars
                    ))
            # Save reviews
            reviews_df = pd.DataFrame(
                reviews_records,
                columns=['user', 'business', 'stars']
            )
            assert reviews_df.stars.min() >= 1
            assert reviews_df.stars.max() <= 5
            assert reviews_df.user.min() >= 0
            assert reviews_df.user.max() < np.iinfo(np.uint32).max
            assert reviews_df.business.min() >= 0
            assert reviews_df.business.max() < np.iinfo(np.uint32).max

            reviews_df.stars = reviews_df.stars.astype(np.uint8)
            reviews_df.user = reviews_df.user.astype(np.uint32)
            reviews_df.business = reviews_df.business.astype(np.uint32)

            rfile = REVIEWS_FILE + '.{}.{}'.format(k, phase)
            to_pickle_and_gz(reviews_df, rfile,
                             csv=args.csv,
                             gzip=args.csv_gzip)

            phase_reviews_sorted = sorted(list(phase_reviews.items()),
                                          key=lambda x: x[0])
            phase_matches = []
            for user, user_reviews in tqdm(phase_reviews_sorted,
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
                    phase_matches.append((uid, r1[0], r2[0], win))

            print("Creating split {} {}".format(k, phase))
            phase_df = pd.DataFrame(phase_matches,
                                    columns=['user', 'b1', 'b2', 'win'])
            # Better dtypes
            assert phase_df.user.min() >= 0
            assert phase_df.user.max() < np.iinfo(np.uint32).max
            assert phase_df.b1.min() >= 0
            assert phase_df.b1.max() < np.iinfo(np.uint32).max
            assert phase_df.b2.min() >= 0
            assert phase_df.b2.max() < np.iinfo(np.uint32).max

            phase_df.user = phase_df.user.astype(np.uint32)
            phase_df.b1 = phase_df.b1.astype(np.uint32)
            phase_df.b2 = phase_df.b2.astype(np.uint32)
            phase_df.win = phase_df.win.astype(np.int8)

            mfile = MATCHES_FILE + '.{}.{}'.format(k, phase)
            print("Saving to {}".format(mfile))
            to_pickle_and_gz(phase_df, mfile,
                             csv=args.csv,
                             gzip=args.csv_gzip)
