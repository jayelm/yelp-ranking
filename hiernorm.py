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
import numpy as np
from scipy.stats import norm
import numpy as np


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

    parser.add_argument('--n_samples', default=2000, type=int,
                        help='Number of MCMC samples per chain')
    parser.add_argument('--thin', default=1, type=int,
                        help='MCMC thinning')
    parser.add_argument('--n_chains', default=4, type=int,
                        help='Number of MCMC chains')
    parser.add_argument('--test', action='store_true',
                        help='Test with fake data')
    parser.add_argument('--save_stan', action='store_true',
                        help='Save Stan model and fit data as pickles')

    parser.add_argument(
        '--save',
        default='results/hiernorm_{n_samples}',
        type=str,
        help='Where to save results (will append .model..pkl and .csv)')

    args = parser.parse_args()

    reviews = pd.read_pickle(args.rfile)

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
    stan_users = reviews.user.apply(lambda u: reindexed_users[u])

    if args.test:
        s1 = 100
        s2 = 50
        s3 = 10
        u1 = norm(3, 1).rvs(size=s1)
        u2 = norm(2, 0.2).rvs(size=s2)
        u3 = norm(4, 0.5).rvs(size=s3)
        data = {
            'n': s1 + s2 + s3,
            'p': 3,
            'y': np.concatenate([u1, u2, u3]),
            'user_idx': np.concatenate(
                [np.repeat(1, s1), np.repeat(2, s2), np.repeat(3, s3)]
            )
        }
    else:
        data = {
            # Number of reviews
            'n': reviews.shape[0],
            # Number of users
            'p': n_users,
            # All reviews
            'y': reviews.stars.values,
            'user_idx': stan_users.values.astype(np.uint32)
        }

    print("Building model")
    with open(MODEL, 'r') as fin:
        model_code = fin.read()
    model = StanModel_cache(model_code=model_code, model_name='hiernorm')
    print("Doing sampling: {} samples x {} chains".format(
        args.n_samples, args.n_chains))
    fit = model.sampling(data=data, iter=args.n_samples, chains=args.n_chains,
                         thin=args.thin)

    if args.save_stan:
        fit_fname = args.save.format(**vars(args)) + '.fit.pkl'
        model_fname = args.save.format(**vars(args)) + '.model.pkl'
        with open(model_fname, 'wb') as mf:
            pickle.dump(model, mf)
        print("Saved", fit_fname)
        with open(fit_fname, 'wb') as ff:
            pickle.dump(fit, ff)
        print("Saved", model_fname)

    feather_fname = args.save.format(**vars(args)) + '.feather'
    fits = fit.summary()
    fits_df = pd.DataFrame(
        fits['summary'],
        columns=fits['summary_colnames'],
        index=fits['summary_rownames'],
        dtype=np.float32
    ).reset_index()
    fits_df.to_feather(feather_fname)
    print("Saved", feather_fname)

    print("Highest Rhat:")
    print(fits_df.Rhat.sort_values(ascending=False).head())
    # Renormalize reviews

    reviews['review_scaled'] = np.nan
    reviews['review_scaled'] = reviews['review_scaled'].astype(np.float32)
    mu_i = fit.extract('mu_i')['mu_i']
    sigma_i = fit.extract('sigma_i')['sigma_i']
    means_only_records = []
    for stan_i, u_i in reindex_rev.items():
        stan_i = stan_i - 1  # Back to numpy, 0 indexed!
        # mean
        u_mean_samps = mu_i[:, stan_i]
        u_mean = u_mean_samps.mean()
        u_mean_std = u_mean_samps.std()

        # stddev
        u_std_samps = sigma_i[:, stan_i]
        u_std = u_std_samps.mean()
        u_std_std = u_std_samps.std()

        # Add to means only records
        means_only_records.append((u_mean, u_mean_std, u_std, u_std_std))

        # Re-scale existing reviews
        user_reviews = reviews[reviews.user == u_i]
        user_zscores = (user_reviews.stars - u_mean) / u_std
        reviews.loc[reviews.user == u_i, 'review_scaled'] = user_zscores

    # Save means only
    mor_feather_fname = args.save.format(**vars(args)) + '.meansonly.feather'
    mor_df = pd.DataFrame(
        means_only_records, columns=['mean', 'mean_std', 'std', 'std_std']
    ).reset_index()
    mor_df.to_feather(mor_feather_fname)
    print("Saved", mor_feather_fname)

    # Save scaled reviews
    reviews_scaled_fname = args.save.format(**vars(args)) + '.reviews.feather'
    reviews.reset_index().to_feather(reviews_scaled_fname)
    print("Saved", reviews_scaled_fname)
