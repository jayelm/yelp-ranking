import pystan
import numpy as np

if __name__ == '__main__':
    data = {
        # Number of reviews
        'n': 10,
        # Number of users
        'p': 2,
        # All reviews
        'y': np.array([5, 4, 3, 4, 4, 1, 2, 1, 4, 3]),
        'user_idx': np.array([1, 1, 1, 1, 1, 2, 2, 1, 1, 1])
    }

    res = pystan.stan('hiernorm.stan', data=data, iter=100)
    print(res)
