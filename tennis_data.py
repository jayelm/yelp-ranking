import requests
import scipy.io
import io
import numpy as np

r = requests.get('https://teachingfiles.blob.core.windows.net/probml/tennis_data.mat')
with io.BytesIO(r.content) as f:
    data = scipy.io.loadmat(f)
    W = np.concatenate(data['W'].squeeze())
    G = data['G'] - 1   # W[G[i,0]] is winner of game i, W[G[i,1]] is loser
    M = W.shape[0]      # number of players M = 107
    N = G.shape[0]      # number of games N = 1801

