from phe import paillier
import numpy as np

np.random.seed(42)

hidden_dim = 20 #otimizar (ta demais)

reg_u = 1e-4
reg_v = 1e-4

lr = 0.01

band_width = 1 # Gb/s

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024, )

dataset = 'ml-32m'

hiperparam = 5 #dar um nome

aggregation_int = 1_000_000

k = 20  #para precision e ndcg