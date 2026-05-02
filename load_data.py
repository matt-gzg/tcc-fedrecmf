import os
import csv
import numpy as np
from collections import defaultdict
import pandas as pd

from shared_parameter import dataset

def load_csv(fileName, fileWithHeader=True):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) if fileWithHeader else []
        data = [r for r in reader]
    return header, data

def load_dat():
    df = pd.read_csv(
    os.path.join(data_path, 'ratings.dat'),
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
    )
    return df.columns.tolist(), df.values.tolist()

if dataset == 'ml-100k':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'ml-latest-small')
    headers, ratings = load_csv(os.path.join(data_path, 'ratings.csv'))
else:
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, dataset)
    headers, ratings = load_dat()
#ratings: [userId, movieId, rating, timestamp]

num_items = len(ratings)
user_ids = set(int(e[0]) for e in ratings)
num_users = len(user_ids)
print(len(ratings))
print(num_users)

#ordena pelo timestamp pra fazer a linha temporal
ratings_sorted_by_time = sorted(ratings, key=lambda x: int(x[-1]))
items_list = ratings_sorted_by_time[:num_items]
item_id_list = sorted(set([int(e[1]) for e in items_list]), key=lambda x:int(x))[:num_items]

#pega a qtdade de users do dataset (na real n ta bem assim)
user_id_list = sorted(set([e[0] for e in ratings]), key=lambda x:int(x))[:num_users]
user_id_set = set(user_id_list)
item_id_map = {item_id: idx for idx, item_id in enumerate(item_id_list)}

ratings_dict = {e: [] for e in user_id_list}
counter = 0
for record in ratings:
    user_id = record[0]
    item_id = int(record[1])
    if user_id not in user_id_set or item_id not in item_id_map:
        continue
    counter += 1
    ratings_dict[user_id].append([item_id_map[item_id], float(record[2]), int(record[3])])

train_ratio = 0.8
train_data = {}
test_data  = {}

for uid in user_id_list:
    sorted_rate = sorted(ratings_dict[uid], key=lambda x: x[-1])
    split_idx = max(1, int(len(sorted_rate) * train_ratio))
    train_data[uid] = sorted_rate[:split_idx]
    test_data[uid]  = sorted_rate[split_idx:]

print('Number of items:', len(items_list))
print('Number of users:', len(user_id_list))
print('Number of training ratings:', sum(len(train_data[u]) for u in train_data))
print('Number of testing  ratings:', sum(len(test_data[u])  for u in test_data))