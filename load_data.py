import os
import pandas as pd
from shared_parameter import dataset

if dataset == 'ml-100k' or dataset == 'ml-32m' or dataset == 'ml-test':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, dataset)
    dtype = {'userId': str, 'movieId': int, 'rating': float, 'timestamp': int}
    df = pd.read_csv(
        os.path.join(data_path, 'ratings.csv'),
        dtype=dtype
    )

#ordena por timestamp e reseta os indices
df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

#pegar ids sem repetir
user_id_list = sorted(df['userId'].unique(), key=lambda x: int(x))
item_id_list = sorted(df['movieId'].unique())

#treseta os indices de 0 a n-1
item_id_map  = {item_id: idx for idx, item_id in enumerate(item_id_list)}

#cria uma nova coluna
df['item_idx'] = df['movieId'].map(item_id_map)

#train/validation/test
train_ratio = 0.8
validation_ratio = 0.2

n = len(df)
train_val_end = max(1, int(n * train_ratio))
train_end     = max(1, int(train_val_end * (1 - validation_ratio)))

train_df      = df.iloc[:train_end]
validation_df = df.iloc[train_end:train_val_end]
test_df       = df.iloc[train_val_end:]

train_data      = {uid: [] for uid in user_id_list}
validation_data = {uid: [] for uid in user_id_list}
test_data       = {uid: [] for uid in user_id_list}

for _, row in train_df.iterrows():
    train_data[row['userId']].append((row['item_idx'], row['rating'], row['timestamp']))

for _, row in validation_df.iterrows():
    validation_data[row['userId']].append((row['item_idx'], row['rating'], row['timestamp']))

for _, row in test_df.iterrows():
    test_data[row['userId']].append((row['item_idx'], row['rating'], row['timestamp']))

global_train = [
    (uid, item_idx, rate, timestamp)
    for uid in user_id_list
    for item_idx, rate, timestamp in train_data[uid]
]

global_train.sort(key=lambda x: x[-1])