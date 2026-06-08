import time
import numpy as np
import logging
from collections import defaultdict

from shared_parameter import *
from load_from_cache import (user_id_list, item_id_list, stream_data)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('training_fed.log', mode='w'),
    ]
)

logger = logging.getLogger(__name__)

def user_update(user_vector, item_id, v):
    gradient = {}
    p_ui = 1.0

    for _ in range(iter):
        error = p_ui - np.dot(user_vector, v)
        user_vector = user_vector - lr * (-2 * error * v + 2 * reg_u * user_vector)
        gradient[item_id] = lr * (-2 * error * user_vector + 2 * reg_v * v)

    return user_vector, gradient

def aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated, item_users_all):
    for item_id in updated_items:
        total_interacs = 0
        weighted_sum = np.zeros(hidden_dim)

        for uid in item_users_updated[item_id]:
            n = interact_count[uid]
            weighted_sum += (n * items_matrix_local[uid][item_id])
            total_interacs += n

        if total_interacs > 0:
            items_matrix_global[item_id] = weighted_sum / total_interacs

            for uid in item_users_all[item_id]:
                items_matrix_local[uid][item_id] = items_matrix_global[item_id].copy()

    return items_matrix_global, items_matrix_local

def main(train_end, validation_end):
    time_dataset = time.perf_counter()

    users_matrix = 0.1 * np.random.randn(len(user_id_list), hidden_dim)
    items_matrix_global = 0.1 * np.random.randn(len(item_id_list), hidden_dim)
    items_matrix_local = {uid: {} for uid in user_id_list}
    items_matrix_global_T = items_matrix_global.T

    user_id_map = {uid: i for i, uid in enumerate(user_id_list)}
    seen_items_online = {uid: set() for uid in user_id_list}
    user_time_list = []
    updated_items = set()
    interact_count = {uid: 0 for uid in user_id_list}
    item_users_updated = {}
    aggregation_time_list = []
    item_users_all = {}

    hits = 0
    total_predictions = 0

    last_aggregation_time = time.perf_counter()
    last_log_time = time.perf_counter()

    # FASE 1: BSGD federado
    logger.info('Iniciando BSGD federado')
    train_data = list(stream_data[:train_end])

    user_train_data = defaultdict(list)
    for uid, item_id in train_data:
        user_train_data[uid].append(item_id)
        if item_id not in item_users_all:
            item_users_all[item_id] = set()
        item_users_all[item_id].add(uid)

    aggregation_int_batch = aggregation_int * iter_b
    batch_obs_count = 0
    batch_updated_items = set()
    batch_interact_count = {uid: 0 for uid in user_id_list}
    batch_item_users_updated = {}
    batch_aggregation_count = 0

    for epoch in range(iter_b):
        for uid, items in user_train_data.items():
            np.random.shuffle(items)
            user_index = user_id_map[uid]

            for item_id in items:
                if item_id not in items_matrix_local[uid]:
                    items_matrix_local[uid][item_id] = items_matrix_global[item_id].copy()

                users_matrix[user_index], gradient = user_update(
                    users_matrix[user_index], item_id, items_matrix_local[uid][item_id]
                )
                items_matrix_local[uid][item_id] -= gradient[item_id]

                batch_updated_items.add(item_id)
                batch_interact_count[uid] += 1

                if item_id not in batch_item_users_updated:
                    batch_item_users_updated[item_id] = set()
                batch_item_users_updated[item_id].add(uid)

                batch_obs_count += 1
                if batch_obs_count % aggregation_int_batch == 0:
                    items_matrix_global, items_matrix_local = aggregate_fedavg(
                        items_matrix_global, items_matrix_local,
                        batch_interact_count, batch_updated_items,
                        batch_item_users_updated, item_users_all
                    )
                    items_matrix_global_T = items_matrix_global.T
                    batch_updated_items = set()
                    batch_interact_count = {uid: 0 for uid in user_id_list}
                    batch_item_users_updated = {}
                    batch_aggregation_count += 1
                    logger.info(f'[BATCH Agg {batch_aggregation_count}] obs={batch_obs_count} | Time={time.perf_counter() - time_dataset:.2f}s')

        logger.info(f'[BATCH] Epoch {epoch + 1}/{iter_b} | Time={time.perf_counter() - time_dataset:.2f}s')

    # agrega restante ao final do batch
    if batch_updated_items:
        items_matrix_global, items_matrix_local = aggregate_fedavg(
            items_matrix_global, items_matrix_local,
            batch_interact_count, batch_updated_items,
            batch_item_users_updated, item_users_all
        )
        items_matrix_global_T = items_matrix_global.T
        batch_aggregation_count += 1
        logger.info(f'[BATCH Agg final {batch_aggregation_count}] Time={time.perf_counter() - time_dataset:.2f}s')

    # popula seen_items após batch
    for uid, item_id in train_data:
        seen_items_online[uid].add(item_id)

    # FASE 2: ISGD federado — test-then-learn incremental
    logger.info('Iniciando ISGD federado')
    for obs_count, (uid, item_id) in enumerate(stream_data[train_end:validation_end], start=1):
        t = time.perf_counter()

        user_index = user_id_map[uid]

        p = np.dot(users_matrix[user_index], items_matrix_global_T)

        if items_matrix_local[uid]:
            local_ids = list(items_matrix_local[uid].keys())
            local_vecs = np.array(list(items_matrix_local[uid].values()))
            p[local_ids] = users_matrix[user_index] @ local_vecs.T

        seen_items = seen_items_online[uid]
        if seen_items:
            p[list(seen_items)] = -np.inf

        proximity = np.abs(1 - p)
        top_k = np.argpartition(proximity, k)[:k]
        if item_id in top_k:
            hits += 1
        total_predictions += 1

        if item_id not in items_matrix_local[uid]:
            items_matrix_local[uid][item_id] = items_matrix_global[item_id].copy()

        users_matrix[user_index], gradient = user_update(
            users_matrix[user_index], item_id, items_matrix_local[uid][item_id]
        )
        items_matrix_local[uid][item_id] -= gradient[item_id]

        updated_items.add(item_id)
        interact_count[uid] += 1

        if item_id not in item_users_updated:
            item_users_updated[item_id] = set()
        item_users_updated[item_id].add(uid)

        seen_items_online[uid].add(item_id)

        if item_id not in item_users_all:
            item_users_all[item_id] = set()
        item_users_all[item_id].add(uid)

        user_time_list.append(time.perf_counter() - t)

        if obs_count % 10000 == 0:
            current_time = time.perf_counter()
            block_time = current_time - last_log_time
            logger.info(
                f'[VALIDATION] Processed={obs_count} | '
                f'AvgUserTime={np.mean(user_time_list[-10000:]):.6f}s | '
                f'BlockTime={block_time:.6f}s | '
                f'TotalTime={time.perf_counter() - time_dataset:.6f}s | '
                f'HR@{k}={hits / total_predictions:.4f}'
            )
            last_log_time = current_time

        if obs_count % aggregation_int == 0:
            aggregation_start = time.perf_counter()

            items_matrix_global, items_matrix_local = aggregate_fedavg(
                items_matrix_global, items_matrix_local,
                interact_count, updated_items, item_users_updated, item_users_all
            )
            items_matrix_global_T = items_matrix_global.T

            aggregation_elapsed = time.perf_counter() - aggregation_start
            aggregation_time_list.append(aggregation_elapsed)

            item_users_updated = {}
            updated_items = set()
            interact_count = {uid: 0 for uid in user_id_list}

            aggregation_time = time.perf_counter() - last_aggregation_time
            logger.info(
                f'[Agg {obs_count // aggregation_int}] '
                f'AggTime={aggregation_elapsed:.6f}s | '
                f'Time={aggregation_time:.2f}s'
            )
            last_aggregation_time = time.perf_counter()

    logger.info(f'HR@{k}: {hits / total_predictions:.4f}')
    logger.info(f'User Average Time: {np.mean(user_time_list):.6f}')
    logger.info(f'Total Aggregations: {len(aggregation_time_list)}')
    logger.info(f'Average Aggregation Time: {np.mean(aggregation_time_list):.6f}s')
    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')

if __name__ == '__main__':
    n = len(stream_data)

    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    logger.info(
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | iter: {iter} | iter_b: {iter_b}\n'
        f'Dataset size: {n} | '
        f'Train: {train_end} | '
        f'Validation: {validation_end - train_end} | '
        f'Test: {n - validation_end}'
    )

    main(train_end, validation_end)