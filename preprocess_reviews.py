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

REVIEW_FILE = 'dataset/review.json'

USERS_FILE = 'dataset_processed/users.pkl'
BUSINESSES_FILE = 'dataset_processed/businesses.pkl'
MATCHES_FILE = 'dataset_processed/matches.pkl'

EDGELISTS_DRAWS = 'dataset_processed/matches_draws_edgelist.csv'
EDGELISTS_NO_DRAWS = 'dataset_processed/matches_no_draws_edgelist.csv'

SENTIMENTS = {
    'very_positive': 4,
    'positive': 3,
    'negative': 2,
    'very_negative': 1
}
SENTIMENTS_NLP_KEY = {k.replace('_', ' '): v for k, v in SENTIMENTS.items()}

Review = namedtuple('Review', ['user_id', 'business_id', 'stars', 'sentiment'])


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
    res = nlp.annotate(
        text,
        properties={
            'annotators': 'sentiment',
            'outputFormat': 'json'
        }
    )
    if isinstance(res, str):
        raise Exception("Got string output: {}".format(res))
    if use_expectation:
        sent_vals = list(map(expectation, [x['sentimentDistribution']
                                           for x in res['sentences']]))
    else:
        sent_vals = list(map(float, [x['sentimentValue']
                                     for x in res['sentences']]))
    mean_sent = sum(sent_vals) / len(sent_vals)
    # Just get average sentiment
    return mean_sent


def review_from_json_str(json_str):
    """
    Create a review object from the JSON, performing sentiment analysis on the
    text.
    """
    review_json = json.loads(json_str)
    sentiment = analyze_sentiment(review_json['text'])
    return Review(review_json['user_id'],
                  review_json['business_id'],
                  review_json['stars'],
                  sentiment)


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
    business_ratings = defaultdict(list)
    business_sents = defaultdict(list)

    n_lines = file_len(REVIEW_FILE)
    print("{} reviews".format(n_lines))

    import multiprocessing as mp

    # Load reviews
    print("Opening reviews")
    with open(REVIEW_FILE, 'r') as rf:
        rf_lines = list(rf)
        #  for line in tqdm(rf, total=n_lines, desc='Loading reviews'):
            #  review = review_from_json_str(line)
            #  all_reviews[review.user_id].append(review)
            #  all_users.add(review.user_id)
            #  business_ratings[review.business_id].append(review.stars)
            #  business_sents[review.business_id].append(review.sentiment)

    print("Sentiment")
    pool = mp.Pool(4)
    reviews = pool.map(review_from_json_str, tqdm(rf_lines, total=n_lines, desc='Loading reviews'))
    for review in reviews:
        all_reviews[review.user_id].append(review)
        all_users.add(review.user_id)
        business_ratings[review.business_id].append(review.stars)
        business_sents[review.business_id].append(review.sentiment)

    print("{} users".format(len(all_users)))
    print("{} businesses".format(len(business_ratings.keys())))

    import ipdb; ipdb.set_trace()
    avg_business_records = [(b, sum(rs) / len(rs), len(rs))
                            for b, rs in business_ratings.items()]
    avg_business_sents = [(b, sum(ss) / len(ss), len(ss))
                          for b, ss in business_sents.items()]
    business_df = pd.DataFrame(
        avg_business_records,
        columns=['business_id', 'avg_rating', 'n_reviews'],
    )
    business_df['avg_sent'] = business_df.business_id.apply(
        lambda bid: sum(business_sents[bid]) / len(business_sents[bid])
    )
    business_sents_counts = {k: Counter(v) for k, v in business_sents.items()}
    for sent_level, i in SENTIMENTS.items():
        business_df[sent_level] = business_df.business_id.apply(
            lambda bid: business_sents_counts[bid][i]
        )
    business_df['avg_sent'] = business_df.business_id.apply(
        lambda bid: sum(business_sents[bid]) / len(business_sents[bid])
    )
    # Smaller dtypes to save space
    assert business_df.n_reviews.min() > 0
    assert business_df.n_reviews.max() < np.iinfo(np.uint16).max

    for sent_level in SENTIMENTS:
        business_df[sent_level] = business_df[sent_level].astype(np.uint16)

    business_df.n_reviews = business_df.n_reviews.astype(np.uint16)
    business_df.avg_rating = business_df.avg_rating.astype(np.float32)
    business_df.avg_sent = business_df.avg_sent.astype(np.float32)
    # Make dict *before* coercing to bytes
    businesses_to_ids = dict(zip(business_df.business_id, business_df.index))
    business_df.business_id = business_df.business_id.astype(np.character)

    to_pickle_and_gz(business_df, BUSINESSES_FILE,
                     csv=args.csv,
                     gzip=args.csv_gzip)

    # Convert users and businesses into integer ids
    users_df = pd.DataFrame(list(all_users),
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
