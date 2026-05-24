import time
import numpy as np

from shared_parameter import *
from load_data import (train_data, test_data, user_id_list, item_id_list, global_train)


def user_update(user_vector, item_id, rate, v):
    gradient = {}
    p_ui = 1.0 if rate >= 4 else 0.0

    for _ in range(hiperparam):
        error = p_ui - np.dot(user_vector, v)
        user_vector = user_vector - lr * (-2 * error * v + 2 * reg_u * user_vector)
        gradient[item_id] = lr * (-2 * error * user_vector + 2 * reg_v * v)

    return user_vector, gradient

def aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated):
    for item_id in updated_items:
        total_ratings = 0
        weighted_sum = np.zeros(hidden_dim)

        for uid in item_users_updated[item_id]:
            n = interact_count[uid]
            weighted_sum += (n * items_matrix_local[uid][item_id])
            total_ratings += n

        if total_ratings > 0:
            items_matrix_global[item_id] = weighted_sum / total_ratings

            for uid in item_users_updated[item_id]:
                items_matrix_local[uid][item_id] = (items_matrix_global[item_id].copy())

    return items_matrix_global, items_matrix_local

def evaluate(user_idx, uid, data, users_matrix, items_matrix_global, items_matrix_local):
    items = data[uid]

    if not items:
        return None, None

    relevant_items = set(item_id for item_id, rate, _ in items if rate >= 4)

    if not relevant_items:
        return None, None

    p = np.dot(users_matrix[user_idx], items_matrix_global.T)

    for iid, local_vec in items_matrix_local[uid].items():
        p[iid] = np.dot(users_matrix[user_idx], local_vec)

    seen_items = set(iid for iid, _, _ in train_data[uid])
    p[list(seen_items)] = -np.inf
    ranked_items = np.argsort(p)[::-1][:k]
    num_relevant = len(relevant_items)

    dcg = 0.0
    idcg = 0.0
    hits = 0

    for rank, item in enumerate(ranked_items):
        rel = 1 if item in relevant_items else 0
        hits += rel
        dcg += rel / np.log2(rank + 2)

        if rank < num_relevant:
            idcg += 1 / np.log2(rank + 2)

    hit_rate = 1.0 if hits > 0 else 0.0
    ndcg = (dcg / idcg if idcg > 0 else 0.0)

    return hit_rate, ndcg

if __name__ == '__main__':
    time_dataset = time.time()
    log_file = open('agg_data.txt', 'w')

    users_matrix = 0.1 * np.random.randn(len(user_id_list), hidden_dim)
    items_matrix_global = 0.1 * np.random.randn(len(item_id_list), hidden_dim)
    items_matrix_local = {uid: {} for uid in user_id_list}

    user_id_map = {uid: i for i, uid in enumerate(user_id_list)}
    user_time_list = []
    prequential_hits = []
    updated_items = set()
    obs_count = 0
    last_aggregation_time = time.time()
    interact_count = {uid: 0 for uid in user_id_list}
    item_users_updated = {}

    for uid, item_id, rating, timestamp in global_train:
        t = time.time()

        obs_count += 1
        interact_count[uid] += 1

        user_index = user_id_map[uid]

        p = np.dot(users_matrix[user_index], items_matrix_global.T)

        for iid, local_vec in items_matrix_local[uid].items():
            p[iid] = np.dot(users_matrix[user_index], local_vec)

        top_k = np.argsort(p)[::-1][:k]
        prequential_hits.append(1 if item_id in top_k else 0)

        if item_id not in items_matrix_local[uid]:
            items_matrix_local[uid][item_id] = (items_matrix_global[item_id].copy())

        current_item = items_matrix_local[uid][item_id]

        users_matrix[user_index], gradient = user_update(users_matrix[user_index], item_id, rating, current_item)
        items_matrix_local[uid][item_id] -= gradient[item_id]
        updated_items.add(item_id)

        if item_id not in item_users_updated:
            item_users_updated[item_id] = set()
        item_users_updated[item_id].add(uid)

        if obs_count % aggregation_int == 0:
            hit_rate = np.mean(prequential_hits[-aggregation_int:])

            items_matrix_global, items_matrix_local = aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated)

            item_users_updated = {}
            updated_items = set()
            interact_count = {uid: 0 for uid in user_id_list}
            aggregation_time = time.time() - last_aggregation_time
            agg = (
                f'[Aggregation #{obs_count // aggregation_int}] '
                f'ratings={obs_count} | '
                f'HR@{k}={hit_rate:.4f} | '
                f'time={aggregation_time:.2f}s'
            )
            print(agg)
            log_file.write(agg + '\n')
            log_file.flush()
            last_aggregation_time = time.time()

        user_time_list.append(time.time() - t)

    final_hr_list = []
    final_ndcg_list = []

    for i, uid in enumerate(user_id_list):
        hr, n = evaluate(i, uid, test_data, users_matrix, items_matrix_global, items_matrix_local)
        if hr is not None:
            final_hr_list.append(hr)
            final_ndcg_list.append(n)

    print(f'Prequencial Hit@{k}: 'f'{np.mean(prequential_hits):.4f}')
    print('HR@{}: {:.4f}'.format(k, np.mean(final_hr_list)))
    print('NDCG@{}: {:.4f}'.format(k, np.mean(final_ndcg_list)))
    print('User Average Time: {:.4f}'.format(np.mean(user_time_list)))
    print('Total Time: {:.4f} seconds'.format(time.time() - time_dataset))