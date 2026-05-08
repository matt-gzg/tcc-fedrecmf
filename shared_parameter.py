from phe import paillier

hidden_dim = 200

max_iteration = 50

reg_u = 1e-4
reg_v = 1e-4

lr = 0.001

band_width = 1 # Gb/s

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024, )

dataset = 'ml-100k'

hiperparam = 50

k = 10  # for precision@k and NDCG@k

num_neg_samples = 4  # number of negative samples