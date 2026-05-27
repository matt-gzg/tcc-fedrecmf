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

train_data = cached_data['train_data']
validation_data = cached_data['validation_data']
test_data = cached_data['test_data']

user_id_list = cached_data['user_id_list']
item_id_list = cached_data['item_id_list']
item_id_map = cached_data['item_id_map']

print('Building global interactions...')

print('Building global_train...')
global_train = [
    (uid, item_idx, timestamp)
    for uid in user_id_list
    for item_idx, timestamp in train_data[uid]
]

print('Building global_validation...')
global_validation = [
    (uid, item_idx, timestamp)
    for uid in user_id_list
    for item_idx, timestamp in validation_data[uid]
]

print('Building global_test...')
global_test = [
    (uid, item_idx, timestamp)
    for uid in user_id_list
    for item_idx, timestamp in test_data[uid]
]

print('Sorting global interactions...')

global_train.sort(key=lambda x: x[-1])
global_validation.sort(key=lambda x: x[-1])
global_test.sort(key=lambda x: x[-1])

num_users = len(user_id_list)
num_items = len(item_id_list)

print(f'Users: {num_users}')
print(f'Items: {num_items}')

print(f'Train interactions: {len(global_train)}')
print(f'Validation interactions: {len(global_validation)}')
print(f'Test interactions: {len(global_test)}')

total_interactions = (
    len(global_train) +
    len(global_validation) +
    len(global_test)
)

print(f'Total interactions: {total_interactions}')

sparsity = 1.0 - (total_interactions / (num_users * num_items))
print(f'Sparsity: {sparsity:.6f}')

print('Dataset loaded.')

print(f'Loading time: {time.perf_counter() - start:.2f}s')