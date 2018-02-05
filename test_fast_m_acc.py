from run_trueskill_mp import gaussian_ep
import numpy as np
from tqdm import trange

from tennis_data import *

G = G.astype(np.uint32)

print("Regular")
print()
num_mp_samples = 80
mp_samples = np.zeros((num_mp_samples, 2, W.shape[0]))
for it, samp in zip(trange(num_mp_samples), gaussian_ep(G, M)):
    mean, stdev = samp
    mp_samples[it, :] = np.array([mean, stdev])

x = mp_samples.copy()

print("Fast")
print()
num_mp_samples = 80
mp_samples = np.zeros((num_mp_samples, 2, W.shape[0]))
for it, samp in zip(trange(num_mp_samples), gaussian_ep(G, M, fast_m_acc=True)):
    mean, stdev = samp
    mp_samples[it, :] = np.array([mean, stdev])

y = mp_samples.copy()

print(np.allclose(x, y))
