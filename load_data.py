import os
import csv
import numpy as np
from collections import defaultdict

def load_csv(fileName, fileWithHeader=True):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) if fileWithHeader else []
        data = [r for r in reader]
    return header, data

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'ml-latest-small')
headers, ratings = load_csv(os.path.join(data_path, 'ratings.csv'))
#ratings: [userId, movieId, rating, timestamp]

num_items = len(ratings)
num_users = 610

#ordena pelo timestamp pra fazer a linha temporal
ratings_sorted_by_time = sorted(ratings, key=lambda x: int(x[-1]))
items_list = ratings_sorted_by_time[:num_items]
item_id_list = sorted(set([int(e[1]) for e in items_list]), key=lambda x:int(x))[:num_items]

#pega a qtdade de users do dataset (610) (na real n ta bem assim)
user_id_list = sorted(set([e[0] for e in ratings]), key=lambda x:int(x))[:num_users]

ratings_dict = {e:[] for e in user_id_list}
counter = 0
for record in ratings:
    if record[0] not in user_id_list or int(record[1]) not in item_id_list:
        continue
    counter += 1
    #int(record[1]) chama a posicao e ai converte massa demais
    ratings_dict[record[0]].append([item_id_list.index(int(record[1])), float(record[2]), int(record[3])])
print(ratings_dict)

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