import time
import numpy as np
import logging

from shared_parameter import *
from load_from_cache import (user_id_list, item_id_list, stream_data)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='w'),
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


def aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated):
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


def main(train_end, validation_end):
    time_dataset = time.perf_counter()

    # Inicialização lazy: vetores criados na primeira interação
    users_matrix = {}
    items_matrix_global = {}
    items_matrix_local = {}

    seen_items_online = {}
    user_time_list = []
    updated_items = set()
    interact_count = {}
    item_users_updated = {}
    aggregation_time_list = []

    hits = 0
    total_predictions = 0

    last_aggregation_time = time.perf_counter()
    last_log_time = time.perf_counter()

    logger.info('Começou o fluxo')
    for obs_count, (uid, item_id) in enumerate(stream_data[:validation_end], start=1):
        t = time.perf_counter()

        phase = "train" if obs_count <= train_end else "validation"

        # Inicializa usuário na primeira aparição
        if uid not in users_matrix:
            users_matrix[uid] = 0.1 * np.random.randn(hidden_dim)
            items_matrix_local[uid] = {}
            seen_items_online[uid] = set()
            interact_count[uid] = 0

        # Inicializa item global na primeira aparição
        if item_id not in items_matrix_global:
            items_matrix_global[item_id] = 0.1 * np.random.randn(hidden_dim)

        if phase == 'validation':
            # Scores sobre todos os itens já vistos no stream
            all_item_ids = list(items_matrix_global.keys())
            all_item_vecs = np.array(list(items_matrix_global.values()))
            p = users_matrix[uid] @ all_item_vecs.T

            # Sobrescreve com vetores locais onde existirem
            if items_matrix_local[uid]:
                for i, iid in enumerate(all_item_ids):
                    if iid in items_matrix_local[uid]:
                        p[i] = np.dot(users_matrix[uid], items_matrix_local[uid][iid])

            # Filtra itens já vistos
            seen_items = seen_items_online[uid]
            if seen_items:
                for i, iid in enumerate(all_item_ids):
                    if iid in seen_items:
                        p[i] = -np.inf

            top_k_indices = np.argpartition(p, -k)[-k:]
            top_k_items = {all_item_ids[i] for i in top_k_indices}
            if item_id in top_k_items:
                hits += 1
            total_predictions += 1

        # Inicializa vetor local do item para este usuário
        if item_id not in items_matrix_local[uid]:
            items_matrix_local[uid][item_id] = items_matrix_global[item_id].copy()

        current_item = items_matrix_local[uid][item_id]

        users_matrix[uid], gradient = user_update(users_matrix[uid], item_id, current_item)
        items_matrix_local[uid][item_id] -= gradient[item_id]

        updated_items.add(item_id)
        interact_count[uid] += 1

        if item_id not in item_users_updated:
            item_users_updated[item_id] = set()
        item_users_updated[item_id].add(uid)

        seen_items_online[uid].add(item_id)
        user_time_list.append(time.perf_counter() - t)

        if obs_count % 10000 == 0:
            current_time = time.perf_counter()
            block_time = current_time - last_log_time
            logger.info(
                f'[{phase.upper()}] Processed={obs_count} | '
                f'AvgUserTime={np.mean(user_time_list[-10000:]):.6f}s | '
                f'BlockTime={block_time:.6f}s | '
                f'TotalTime={time.perf_counter() - time_dataset:.6f}s | '
                f"HR@{k}={'N/A' if phase == 'train' else f'{hits / total_predictions:.4f}'}"
            )
            last_log_time = current_time

        if obs_count % aggregation_int == 0:
            aggregation_start = time.perf_counter()

            items_matrix_global, items_matrix_local = aggregate_fedavg(
                items_matrix_global, items_matrix_local, interact_count, updated_items, item_users_updated
            )

            aggregation_elapsed = time.perf_counter() - aggregation_start
            aggregation_time_list.append(aggregation_elapsed)

            item_users_updated = {}
            updated_items = set()
            interact_count = {uid: 0 for uid in users_matrix}

            aggregation_time = time.perf_counter() - last_aggregation_time
            logger.info(
                f'[Agg {obs_count // aggregation_int}] '
                f'AggTime={aggregation_elapsed:.6f}s | '
                f'Time={aggregation_time:.2f}s'
            )
            last_aggregation_time = time.perf_counter()

    logger.info(f'HR@{k}: {hits / total_predictions:.4f} ({total_predictions} predictions)')
    logger.info(f'User Average Time: {np.mean(user_time_list):.6f}')
    logger.info(f'Total Aggregations: {len(aggregation_time_list)}')
    logger.info(f'Average Aggregation Time: {np.mean(aggregation_time_list):.6f}s')
    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')


if __name__ == '__main__':
    n = len(stream_data)

    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    logger.info(
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | iter: {iter} \n'
        f'Dataset size: {n} | '
        f'Train: {train_end} | '
        f'Validation: {validation_end - train_end} | '
        f'Test: {n - validation_end}'
    )

    main(train_end, validation_end)