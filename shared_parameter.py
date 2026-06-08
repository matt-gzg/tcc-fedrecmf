from phe import paillier
import numpy as np

np.random.seed(42)

hidden_dim = [40, 50, 60, 70, 100]
reg_u = [0.001]
reg_v = [0.001]
lr = [0.01]
iter = [1]
iter_b = [4, 6, 8, 10, 15, 20]

band_width = 1 # Gb/s

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024, )

dataset = 'ml-32m'

aggregation_int = 45_965

k = 20  #para precision e ndcg

train_ratio = 0.1
validation_ratio = 0.1
test_ratio = 0.8
sample_ratio = 1.0 #para amostragem
