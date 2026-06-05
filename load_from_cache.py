import os
import pickle
import time

from shared_parameter import dataset

start = time.perf_counter()

current_path = os.path.dirname(os.path.abspath(__file__))

cache_path = os.path.join(
    current_path,
    f'{dataset}_cache.pkl'
)

print('Loading cached dataset...')

with open(cache_path, 'rb') as f:
    cached_data = pickle.load(f)

stream_data = cached_data['stream_data']

user_id_list = cached_data['user_id_list']
item_id_list = cached_data['item_id_list']
item_id_map = cached_data['item_id_map']

num_users = len(user_id_list)
num_items = len(item_id_list)

print(f'Users: {num_users}')
print(f'Items: {num_items}')
print(f'Interactions: {len(stream_data)}')

sparsity = 1.0 - (len(stream_data) / (num_users * num_items))

print(f'Sparsity: {sparsity:.6f}')

print('Dataset loaded.')
print(f'Loading time: {time.perf_counter() - start:.2f}s')