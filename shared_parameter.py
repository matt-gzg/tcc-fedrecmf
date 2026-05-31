from phe import paillier
import numpy as np

np.random.seed(42)

hidden_dim = [50, 100, 150, 200]

reg_u = [1e-4, 1e-3]
reg_v = [1e-4, 1e-3]

lr = [0.3, 0.1, 0.03, 0.01]

band_width = 1 # Gb/s

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024, )

dataset = 'ml-32m'

iter = [5, 10, 25, 50]

aggregation_int = 50_000

k = 20  #para precision e ndcg

sample_ratio = 0.10 #para amostragem

warm_ratio = 0.0