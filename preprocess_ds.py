import os
import pickle
import pandas as pd
import time

from shared_parameter import dataset

start = time.perf_counter()

print('Loading dataset...')

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, dataset)

dtype = {
    'userId': str,
    'movieId': int,
    'rating': float,
    'timestamp': int
}

df = pd.read_csv(
    os.path.join(data_path, 'ratings.csv'),
    dtype=dtype
)

print('Sorting interactions...')

df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

print('Filtering positive interactions...')

df = df[df['rating'] >= 4].copy()

print(f'Remaining interactions: {len(df)}')

print('Removing rating column...')

df = df[['userId', 'movieId', 'timestamp']]

print('Building mappings...')

user_id_list = sorted(df['userId'].unique(), key=lambda x: int(x))
item_id_list = sorted(df['movieId'].unique())

item_id_map = {
    item_id: idx
    for idx, item_id in enumerate(item_id_list)
}

df['item_idx'] = df['movieId'].map(item_id_map)

train_ratio = 0.8
validation_ratio = 0.2

n = len(df)

train_val_end = max(1, int(n * train_ratio))
train_end = max(1, int(train_val_end * (1 - validation_ratio)))

train_df = df.iloc[:train_end]
validation_df = df.iloc[train_end:train_val_end]
test_df = df.iloc[train_val_end:]

print('Building dictionaries...')

train_data = {uid: [] for uid in user_id_list}
validation_data = {uid: [] for uid in user_id_list}
test_data = {uid: [] for uid in user_id_list}

for row in train_df.itertuples(index=False):
    train_data[row.userId].append((row.item_idx, row.timestamp))

for row in validation_df.itertuples(index=False):
    validation_data[row.userId].append((row.item_idx, row.timestamp))

for row in test_df.itertuples(index=False):
    test_data[row.userId].append((row.item_idx, row.timestamp))

del train_df
del validation_df
del test_df
del df

cache_path = os.path.join(current_path, f'{dataset}_cache.pkl')

print('Saving cache...')

with open(cache_path, 'wb') as f:
    pickle.dump({
        'train_data': train_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'user_id_list': user_id_list,
        'item_id_list': item_id_list,
        'item_id_map': item_id_map
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')
print(f'Preprocess time: {time.perf_counter() - start:.2f}s')