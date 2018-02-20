"""
Preprocess dataset/review.json
"""

from collections import defaultdict, namedtuple, Counter
import json
from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np
from pycorenlp import StanfordCoreNLP
import random
import time
import multiprocessing as mp

N_CPU = 12

random.seed(0)

REVIEW_FILE = 'dataset/review.json'

USERS_FILE = 'dataset_processed/users.pkl'
BUSINESSES_FILE = 'dataset_processed/businesses.pkl'
MATCHES_FILE = 'dataset_processed/matches.pkl'

EDGELISTS_DRAWS = 'dataset_processed/matches_draws_edgelist.csv'
EDGELISTS_NO_DRAWS = 'dataset_processed/matches_no_draws_edgelist.csv'

Review = namedtuple('Review', ['user_id', 'business_id', 'stars', 'text', 'sentiment'])


def to_pickle_and_gz(df, fname, csv=False, gzip=False):
    if gzip and not csv:
        raise ValueError("Cannot set gzip without csv=True")
    df.to_pickle(fname)
    if csv:
        # Get rid of byte strings
        for column in df.columns:
            if df[column].dtype.kind == 'S':
                df[column] = df[column].astype(str)
        df.to_csv(fname.replace('.pkl', '.csv.gz'),
                  compression='gzip' if gzip else None)


def expectation(sentiment_distribution):
    total = 0
    for i, s in enumerate(sentiment_distribution):
        total += i * s
    return total


def analyze_sentiment(text, nlp_=None, use_expectation=True):
    """
    Analyze sentiment with PyCoreNLP.

    If expectation=True, then use the expectation of the sentiment
    distribution, rather than the MAP estimate.
    """
    if nlp_ is None:
        try:
            nlp_ = nlp
        except NameError as e:
            print("NLP not found")
            raise e
    res = ''
    attempts = 0
    while isinstance(res, str):
        attempts += 1
        if res != '':  # ( and res is a str)
            if attempts >= 3:
                print("Too many attempts for {}, returning mean sentiment".format(text[:50]))
                # Return neutral sentiment
                return 2.0
            print("Got timeout: {}... [retrying in 5s]".format(text[:50]))
            time.sleep(5)
        res = nlp.annotate(
            text,
            properties={
                'annotators': 'sentiment',
                'outputFormat': 'json'
            }
        )
    if use_expectation:
        sent_vals = list(map(expectation, [x['sentimentDistribution']
                                           for x in res['sentences']]))
    else:
        sent_vals = list(map(float, [x['sentimentValue']
                                     for x in res['sentences']]))
    mean_sent = sum(sent_vals) / len(sent_vals)
    # Just get average sentiment
    return mean_sent


def review_from_json_str(json_str, sentiment=False):
    """
    Create a review object from the JSON, performing sentiment analysis on the
    text.
    """
    review_json = json.loads(json_str)
    if sentiment:
        sent = analyze_sentiment(review_json['text'])
    else:
        sent = None
    return Review(review_json['user_id'],
                  review_json['business_id'],
                  review_json['stars'],
                  review_json['text'],
                  sent)

def add_sentiment(review):
    return Review(
        review.user_id,
        review.business_id,
        review.stars,
        review.text,
        analyze_sentiment(review.text)
    )


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
    parser.add_argument('--corenlp_server', default='http://localhost:9000',
                        help='URL to StanfordCoreNLP server')

    parser.add_argument('--csv', action='store_true',
                        help='Save .csv files')
    parser.add_argument('--csv_gzip', action='store_true',
                        help='Gzipped .csv files (takes a while)')
    parser.add_argument('--save_graphs', action='store_true',
                        help='Save graph edgelists')

    args = parser.parse_args()

    if args.csv_gzip and not args.csv:
        parser.error("Cannot specify --csv_gzip without --csv")

    nlp = StanfordCoreNLP(args.corenlp_server)

    all_reviews = defaultdict(list)
    all_users = set()
    all_businesses = set()
    business_ratings = defaultdict(list)
    business_sents = defaultdict(list)

    n_lines = file_len(REVIEW_FILE)
    print("{} reviews".format(n_lines))

    # Load reviews
    print("Opening reviews")
    with open(REVIEW_FILE, 'r') as rf:
        rf_lines = list(rf)

    # Generate review_objs, without sentiment
    pool = mp.Pool(N_CPU)
    review_objs = pool.map(
        review_from_json_str,
        tqdm(rf_lines, total=n_lines, desc='Loading reviews'))
    pool.close()
    pool.join()

    # Collect businesses
    for review in review_objs:
        all_businesses.add(review.business_id)
    all_businesses = list(all_businesses)

    # Choose a random sample of businesses
    chosen = random.sample(all_businesses, 10000)
    assert len(set(chosen)) == len(chosen)
    chosen = set(chosen)

    # Perform sentiment analyss on a couple of those reviews
    yes_sentiment = []
    no_sentiment = []
    for review in review_objs:
        if review.business_id in chosen:
            yes_sentiment.append(review)
        else:
            no_sentiment.append(review)

    pool = mp.Pool(N_CPU)
    start = time.time()
    yes_sentiment_added = pool.map(
        add_sentiment,
        tqdm(yes_sentiment, desc='Sentiment analysis'))
    print("Elapsed time: {}".format(time.time() - start))

    reviews = yes_sentiment_added + no_sentiment

    for review in reviews:
        all_reviews[review.user_id].append(review)
        all_users.add(review.user_id)
        business_ratings[review.business_id].append(review.stars)
        if review.sentiment is not None:
            business_sents[review.business_id].append(review.sentiment)

    print("{} users".format(len(all_users)))
    print("{} businesses".format(len(business_ratings.keys())))

    avg_business_records = [(b, sum(rs) / len(rs), len(rs))
                            for b, rs in business_ratings.items()]
    avg_business_records = sorted(avg_business_records, key=lambda x: x[0])
    business_df = pd.DataFrame(
        avg_business_records,
        columns=['business_id', 'avg_rating', 'n_reviews'],
    )

    def maybe_business_sent(bid):
        if bid in business_sents:
            return sum(business_sents[bid]) / len(business_sents[bid])
        else:
            return None

    def maybe_var(bid):
        if bid in business_sents:
            return np.var(business_sents[bid])
        else:
            return None

    business_df['avg_sent'] = business_df.business_id.apply(
        maybe_business_sent)
    business_df['sent_var'] = business_df.business_id.apply(
        maybe_var)

    # Smaller dtypes to save space
    assert business_df.n_reviews.min() > 0
    assert business_df.n_reviews.max() < np.iinfo(np.uint16).max

    business_df.n_reviews = business_df.n_reviews.astype(np.uint16)
    business_df.avg_rating = business_df.avg_rating.astype(np.float32)
    business_df.avg_sent = business_df.avg_sent.astype(np.float32)
    business_df.sent_var = business_df.sent_var.astype(np.float32)
    # Make dict *before* coercing to bytes
    businesses_to_ids = dict(zip(business_df.business_id, business_df.index))
    business_df.business_id = business_df.business_id.astype(np.character)

    to_pickle_and_gz(business_df, BUSINESSES_FILE,
                     csv=args.csv,
                     gzip=args.csv_gzip)

    # Convert users and businesses into integer ids
    users_df = pd.DataFrame(sorted(list(all_users)),
                            columns=['user_id'])
    # Make dict *before* coercing to bytes
    users_to_ids = dict(zip(users_df.user_id, users_df.index))
    users_df.user_id = users_df.user_id.astype(np.character)
    to_pickle_and_gz(users_df, USERS_FILE,
                     csv=args.csv,
                     gzip=args.csv_gzip)

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
    assert matches_df.user.min() >= 0
    assert matches_df.user.max() < np.iinfo(np.uint32).max
    assert matches_df.b1.min() >= 0
    assert matches_df.b1.max() < np.iinfo(np.uint32).max
    assert matches_df.b2.min() >= 0
    assert matches_df.b2.max() < np.iinfo(np.uint32).max

    matches_df.user = matches_df.user.astype(np.uint32)
    matches_df.b1 = matches_df.b1.astype(np.uint32)
    matches_df.b2 = matches_df.b2.astype(np.uint32)
    matches_df.win = matches_df.win.astype(np.int8)

    print("Saving to file")
    to_pickle_and_gz(matches_df, MATCHES_FILE,
                     csv=args.csv,
                     gzip=args.csv_gzip)

    if args.save_graphs:
        matches_df = matches_df.rename(
            columns={'b1': 'Source', 'b2': 'Target'}
        )
        print("Saving edgelists")
        matches_df.drop(columns=['user', 'win']).to_csv(
            EDGELISTS_DRAWS, sep=' ',
            header=True, index=False)
        matches_df = matches_df[matches_df.win != 0]
        matches_df.drop(columns=['user', 'win']).to_csv(
            EDGELISTS_NO_DRAWS, sep=' ',
            header=True, index=False)
