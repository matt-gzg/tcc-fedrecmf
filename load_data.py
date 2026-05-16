import os
import pandas as pd
from shared_parameter import dataset

if dataset == 'ml-100k' or dataset == 'ml-32m':
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

user_id_set  = set(user_id_list)
#treseta os indices de 0 a n-1
item_id_map  = {item_id: idx for idx, item_id in enumerate(item_id_list)}

#cria uma nova coluna
df['item_idx'] = df['movieId'].map(item_id_map)

#faz o dicionario em tuplas
ratings_dict = {}
for uid, group in df.groupby('userId', sort=False):
    ratings_dict[uid] = list(
        zip(group['item_idx'], group['rating'], group['timestamp'])
    )
#(item_idx, rating, timestamp)

#train/validation/test
train_ratio = 0.8
validation_ratio = 0.2
train_data = {}
validation_data = {}
test_data  = {}

for uid in user_id_list:
    entries = ratings_dict[uid]
    n = len(entries)
    train_val_end = max(1, int(n * train_ratio))
    train_end = max(1, int(train_val_end * (1 - validation_ratio)))  #20% do treino para validacao
    train_data[uid] = entries[:train_end]
    validation_data[uid] = entries[train_end:train_val_end]
    test_data[uid] = entries[train_val_end:]

global_train = [
    (uid, item_idx, rate, timestamp)
    for uid in user_id_list
    for item_idx, rate, timestamp in train_data[uid]
]

global_train.sort(key=lambda x: x[-1])