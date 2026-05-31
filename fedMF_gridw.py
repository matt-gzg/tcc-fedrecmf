import time
import numpy as np
import logging

from shared_parameter import *
#from load_data import (train_data, test_data, user_id_list, item_id_list, global_train)
from load_from_cache import (train_data, validation_data, test_data, user_id_list, item_id_list, global_train, global_validation, global_test)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='a'),
    ]
)

logger = logging.getLogger(__name__)

def user_update(user_vector, item_id, v, lr, reg_u, reg_v, iter):
    gradient = {}
    p_ui = 1.0

    for _ in range(iter):
        error = p_ui - np.dot(user_vector, v)
        user_vector = user_vector - lr * (-2 * error * v + 2 * reg_u * user_vector)
        gradient[item_id] = lr * (-2 * error * user_vector + 2 * reg_v * v)

    return user_vector, gradient

def aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, hidden_dim):
    for item_id in updated_items:
        total_interacs = 0
        weighted_sum = np.zeros(hidden_dim)

        for uid in item_users_updated[item_id]:
            n = interact_count[uid]
            weighted_sum += (n * items_matrix_local[uid][item_id])
            total_interacs += n

        if total_interacs > 0:
            items_matrix_global[item_id] = weighted_sum / total_interacs

            for uid in item_users_updated[item_id]:
                items_matrix_local[uid][item_id] = items_matrix_global[item_id].copy()

    return items_matrix_global, items_matrix_local

def evaluate(user_idx, uid, data, users_matrix, items_matrix_global, items_matrix_local, seen_items_online):
    items = data[uid]

    if not items:
        return None, None, None, None

    relevant_items = set(item_id for item_id, _ in items)

    if not relevant_items:
        return None, None, None, None

    p = np.dot(users_matrix[user_idx], items_matrix_global.T)

    for iid, local_vec in items_matrix_local[uid].items():
        p[iid] = np.dot(users_matrix[user_idx], local_vec)

    seen_items = seen_items_online[uid]
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
    recall = hits / num_relevant if num_relevant > 0 else 0.0
    ndcg_ai = dcg / idcg if idcg > 0 else 0.0
    ndcg_ap = dcg / num_relevant if num_relevant > 0 else 0.0

    return hit_rate, recall, ndcg_ai, ndcg_ap

def process_interaction(uid, item_id, user_id_map, users_matrix, items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, lr, reg_u, reg_v, iter):
    interact_count[uid] += 1
    user_index = user_id_map[uid]

    if item_id not in items_matrix_local[uid]:
            items_matrix_local[uid][item_id] = (items_matrix_global[item_id].copy())

    current_item = items_matrix_local[uid][item_id]

    users_matrix[user_index], gradient = user_update(users_matrix[user_index], item_id, current_item, lr, reg_u, reg_v, iter)
    items_matrix_local[uid][item_id] -= gradient[item_id]
    updated_items.add(item_id)

    if item_id not in item_users_updated:
        item_users_updated[item_id] = set()
    item_users_updated[item_id].add(uid)

def warm_up(global_train, warm_ratio, user_id_map, users_matrix, items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, seen_items_online, lr, reg_u, reg_v, iter):
    if warm_ratio <= 0:
        return 0

    warm_size = int(len(global_train) * warm_ratio)

    logger.info(
        f'Warm-up started ({warm_size} interactions)'
    )

    for uid, item_id, _ in global_train[:warm_size]:
        process_interaction(uid, item_id, user_id_map, users_matrix, items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, lr, reg_u, reg_v, iter)
        seen_items_online[uid].add(item_id)

    logger.info('Warm-up finished')

    return warm_size

def main(hidden_dim, reg_u, reg_v, lr, iter):
    time_dataset = time.perf_counter()

    users_matrix = 0.1 * np.random.randn(len(user_id_list), hidden_dim)
    items_matrix_global = 0.1 * np.random.randn(len(item_id_list), hidden_dim)
    items_matrix_local = {uid: {} for uid in user_id_list}

    user_id_map = {uid: i for i, uid in enumerate(user_id_list)}
    seen_items_online = {uid: set() for uid in user_id_list}
    user_time_list = []
    prequential_hits = []
    updated_items = set()
    obs_count = 0
    last_aggregation_time = time.perf_counter()
    interact_count = {uid: 0 for uid in user_id_list}
    item_users_updated = {}
    aggregation_time_list = []
    last_log_time = time.perf_counter()

    warm_size = warm_up(global_train, warm_ratio, user_id_map, users_matrix, items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, seen_items_online, lr, reg_u, reg_v, iter)
    if warm_size > 0:
        items_matrix_global, items_matrix_local = aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, hidden_dim)

        updated_items = set()
        item_users_updated = {}
        interact_count = {uid: 0 for uid in user_id_list}

    for uid, item_id, _ in global_train[warm_size:]:
        t = time.perf_counter()

        obs_count += 1
        user_index = user_id_map[uid]

        p = np.dot(users_matrix[user_index], items_matrix_global.T)

        for iid, local_vec in items_matrix_local[uid].items():
            p[iid] = np.dot(users_matrix[user_index], local_vec)

        seen_items = seen_items_online[uid]
        if seen_items:
            p[list(seen_items)] = -np.inf

        top_k = np.argpartition(p, -k)[-k:]
        prequential_hits.append(1 if item_id in top_k else 0)

        process_interaction(uid, item_id, user_id_map, users_matrix, items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, lr, reg_u, reg_v, iter)
        seen_items_online[uid].add(item_id)

        user_time_list.append(time.perf_counter() - t)

        if obs_count % 10000 == 0:
            current_time = time.perf_counter()
            block_time = (current_time - last_log_time)
            logger.info(
                f'Processed={obs_count} | '
                f'AvgUserTime={np.mean(user_time_list[-10000:]):.6f}s | '
                f'BlockTime={block_time:.6f}s | '
                f'TotalTime={time.perf_counter() - time_dataset:.6f}s | '
                f'PreqHR@{k}={np.mean(prequential_hits[-10000:]):.4f}'
            )
            last_log_time = current_time   

        if obs_count % aggregation_int == 0:
            aggregation_start = time.perf_counter()
            hit_rate = np.mean(prequential_hits[-aggregation_int:])

            items_matrix_global, items_matrix_local = aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, hidden_dim)

            aggregation_elapsed = time.perf_counter() - aggregation_start
            aggregation_time_list.append(aggregation_elapsed)

            item_users_updated = {}
            updated_items = set()
            interact_count = {uid: 0 for uid in user_id_list}

            aggregation_time = time.perf_counter() - last_aggregation_time
            throughput = aggregation_int / aggregation_time

            logger.info(
                f'[Agg {obs_count // aggregation_int}] '
                f'HR@{k}={hit_rate:.4f} | '
                f'AggTime={aggregation_elapsed:.6f}s | '
                f'Throughput={throughput:.2f} interactions/s | '
                f'Time={aggregation_time:.2f}s'
            )
            last_aggregation_time = time.perf_counter()

    final_hr_list = []
    final_ndcg_ai_list = []
    final_ndcg_ap_list = []
    final_recall_list = []

    for i, uid in enumerate(user_id_list):
        hr, recall, ndcg_ai, ndcg_ap = evaluate(i, uid, test_data, users_matrix, items_matrix_global, items_matrix_local, seen_items_online)
        if hr is not None:
            final_hr_list.append(hr)
            final_recall_list.append(recall)
            final_ndcg_ai_list.append(ndcg_ai)
            final_ndcg_ap_list.append(ndcg_ap)

    logger.info(f'FINAL | hidden_dim={hidden_dim} | '
    f'reg_u={reg_u} | reg_v={reg_v} | '
    f'lr={lr} | iter={iter} | ')
    logger.info(f'Prequential Hit@{k}: {np.mean(prequential_hits):.4f}')
    logger.info(f'HR@{k}: {np.mean(final_hr_list):.4f}')
    logger.info(f'Recall@{k}: {np.mean(final_recall_list):.4f}')
    logger.info(f'NDCG@{k} (ai): {np.mean(final_ndcg_ai_list):.4f}')
    logger.info(f'NDCG@{k} (ap): {np.mean(final_ndcg_ap_list):.4f}')
    logger.info(f'User Average Time: {np.mean(user_time_list):.6f}')
    logger.info(f'Total Aggregations: {len(aggregation_time_list)}')
    logger.info(f'Average Aggregation Time: {np.mean(aggregation_time_list):.6f}s')
    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')

for h in hidden_dim:
    for ru in reg_u:
        for rv in reg_v:
            for l in lr:
                for it in iter:
                    logger.info(
                        f'Running with hidden_dim={h}, '
                        f'reg_u={ru}, reg_v={rv}, '
                        f'lr={l}, iter={it}'
                    )
                    main(h, ru, rv, l, it)
