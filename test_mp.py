from run_trueskill_mp import gaussian_ep, gaussian_ep_mp
import requests
import scipy.io
import numpy as np
import io
from tqdm import trange


r = requests.get('https://teachingfiles.blob.core.windows.net/probml/tennis_data.mat')
with io.BytesIO(r.content) as f:
    data = scipy.io.loadmat(f)
    W = np.concatenate(data['W'].squeeze())
    G = data['G'] - 1   # W[G[i,0]] is winner of game i, W[G[i,1]] is loser
    M = W.shape[0]      # number of players M = 107
    N = G.shape[0]      # number of games N = 1801

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
