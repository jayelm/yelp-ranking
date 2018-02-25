"""
Compute standardized estimates of user reviews using a Bayesian hierarchical
model.

TODO: There's most likely a way to model the business scores in this rating
too, but unclear if it'll really add any more than just modeling user scores by
itself (you might address that for example...it would really just be like
adding a business prior)

There are advantages of not having that model too - it's more easy to interpret
and process the standard scores.
"""

import pystan
import pandas as pd
from preprocess_reviews import REVIEWS_FILE
import pickle
from hashlib import md5


def StanModel_cache(model_code, model_name=None, **kwargs):
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm


MODEL = 'hiernorm.stan'


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Trueskill message passing',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--rfile', default=REVIEWS_FILE,
                        help='Reviews file to load')

    parser.add_argument('--num_samples', default=100000, type=int,
                        help='Number of NUTS samples')

    parser.add_argument(
        '--save',
        default='results/hiernorm_{num_samples}.pkl',
        type=str,
        help='Where to save')

    args = parser.parse_args()

    reviews = pd.read_pickle(args.rfile)
    reviews = reviews[(reviews.user == 0) | (reviews.user == 2)]

    n_users = len(reviews.user.unique())
    print("{} users".format(n_users))
    print("{} reviews".format(reviews.shape[0]))

    # Make consecutive 1-indexed user ids for Stan
    reindexed_users = {}
    i = 1
    reviews = reviews.sort_values('user')
    for u in reviews.user:
        if u not in reindexed_users:
            reindexed_users[u] = i
            i += 1
    reindex_rev = {v: k for k, v in reindexed_users.items()}

    # Reindex reviews
    reviews.user = reviews.user.apply(lambda u: reindexed_users[u])

    data = {
        # Number of reviews
        'n': reviews.shape[0],
        # Number of users
        'p': n_users,
        # All reviews
        'y': reviews.stars.values,
        'user_idx': reviews.user.values
    }

    with open(MODEL, 'r') as fin:
        model_code = fin.read()
    model = StanModel_cache(model_code=model_code, model_name='hiernorm')
    fit = model.sampling(data=data, iter=100000, chains=4)
    print(fit)
