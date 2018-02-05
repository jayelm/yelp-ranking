from run_trueskill_mp import gaussian_ep, gaussian_ep_mp
import numpy as np
from tqdm import trange

from tennis_data import *

print("Single")
num_mp_samples = 80
mp_samples = np.zeros((num_mp_samples, 2, W.shape[0]))
for it, samp in zip(trange(num_mp_samples), gaussian_ep(G, M)):
    mean, stdev = samp
    mp_samples[it, :] = np.array([mean, stdev])

x = mp_samples[-1].copy()

print("Double")
num_mp_samples = 80
mp_samples = np.zeros((num_mp_samples, 2, W.shape[0]))
for it, samp in zip(trange(num_mp_samples), gaussian_ep_mp(G, M, 4)):
    mean, stdev = samp
    mp_samples[it, :] = np.array([mean, stdev])

y = mp_samples[-1].copy()

print(np.allclose(x, y))