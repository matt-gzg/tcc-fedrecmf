import os
import pickle
import pandas as pd
import time

from shared_parameter import dataset, sample_ratio

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

df = df[df['rating'] == 5].copy()

print(f'Remaining interactions: {len(df)}')

# print(f'Using only {sample_ratio*100:.0f} % of dataset...')

# sample_size = max(1, int(len(df) * sample_ratio))
# df = df.iloc[:sample_size].copy()

# print(f'Using {sample_size} interactions '
#     f'({sample_ratio * 100:.0f}% of filtered dataset)'
# )

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

print('Building dictionaries...')

stream_data = []

for row in df.itertuples(index=False):
    stream_data.append((row.userId, row.item_idx))

del df

cache_path = os.path.join(current_path, f'{dataset}_cache.pkl')

print('Saving cache...')

with open(cache_path, 'wb') as f:
    pickle.dump({
        'stream_data': stream_data,
        'user_id_list': user_id_list,
        'item_id_list': item_id_list,
        'item_id_map': item_id_map
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')
print(f'Preprocess time: {time.perf_counter() - start:.2f}s')