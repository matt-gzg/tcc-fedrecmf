import time
import numpy as np
import logging
from shared_parameter import *
from load_from_cache import (user_id_list, item_id_list, stream_data)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('training_centralized.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

def user_item_update(user_vector, item_vector):
    p_ui = 1.0
    for _ in range(iter):
        error = p_ui - np.dot(user_vector, item_vector)
        grad_user = -2 * error * item_vector + 2 * reg_u * user_vector
        grad_item = -2 * error * user_vector + 2 * reg_v * item_vector
        user_vector = user_vector - lr * grad_user
        item_vector = item_vector - lr * grad_item
    return user_vector, item_vector

def main(train_end, validation_end):
    np.random.seed(42)
    time_dataset = time.perf_counter()

    users_matrix = 0.1 * np.random.randn(len(user_id_list), hidden_dim)
    items_matrix = 0.1 * np.random.randn(len(item_id_list), hidden_dim)

    user_id_map = {uid: i for i, uid in enumerate(user_id_list)}
    seen_items_online = {uid: set() for uid in user_id_list}
    user_time_list = []
    hits = 0
    total_predictions = 0

    # FASE 1: BSGD — treino em batch com shuffle
    train_data = list(stream_data[:train_end])
    for epoch in range(iter_b):
        np.random.shuffle(train_data)
        for uid, item_id in train_data:
            user_index = user_id_map[uid]
            users_matrix[user_index], items_matrix[item_id] = user_item_update(
                users_matrix[user_index], items_matrix[item_id]
            )
        logger.info(f'[BATCH] Epoch {epoch + 1}/{iter_b} | Time={time.perf_counter() - time_dataset:.2f}s')

    logger.info(
    f'User norm mean: {np.mean(np.linalg.norm(users_matrix, axis=1)):.6f}'
    )

    logger.info(
        f'User norm std: {np.std(np.linalg.norm(users_matrix, axis=1)):.6f}'
    )

    logger.info(
        f'Item norm mean: {np.mean(np.linalg.norm(items_matrix, axis=1)):.6f}'
    )

    logger.info(
        f'Item norm std: {np.std(np.linalg.norm(items_matrix, axis=1)):.6f}'
    )

    for uid, item_id in train_data:
        seen_items_online[uid].add(item_id)

    # FASE 2: ISGD — test-then-learn incremental
    items_matrix_T = items_matrix.T
    last_log_time = time.perf_counter()

    for obs_count, (uid, item_id) in enumerate(stream_data[train_end:validation_end], start=1):
        t = time.perf_counter()
        user_index = user_id_map[uid]

        p = np.dot(users_matrix[user_index], items_matrix_T)
        seen_items = seen_items_online[uid]
        if seen_items:
            p[list(seen_items)] = -np.inf

        top_k = np.argpartition(-p, k)[:k]
        if item_id in top_k:
            hits += 1
        total_predictions += 1

        users_matrix[user_index], items_matrix[item_id] = user_item_update(
            users_matrix[user_index], items_matrix[item_id]
        )
        items_matrix_T = items_matrix.T
        seen_items_online[uid].add(item_id)
        user_time_list.append(time.perf_counter() - t)

        if obs_count % 10000 == 0:
            current_time = time.perf_counter()
            logger.info(
                f'[VALIDATION] Processed={obs_count} | '
                f'AvgUserTime={np.mean(user_time_list[-10000:]):.6f}s | '
                f'BlockTime={current_time - last_log_time:.6f}s | '
                f'TotalTime={time.perf_counter() - time_dataset:.6f}s | '
                f'HR@{k}={hits / total_predictions:.4f}'
            )
            last_log_time = current_time

    logger.info(f'HR@{k}: {hits / total_predictions:.4f}')
    logger.info(f'User Average Time: {np.mean(user_time_list):.6f}')
    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')

if __name__ == '__main__':
    n = len(stream_data)

    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    logger.info(
        f'\n'
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | iter: {iter} | iter_b: {iter_b}\n'
        f'Dataset size: {n} | '
        f'Train: {train_end} | '
        f'Validation: {validation_end - train_end} | '
        f'Test: {n - validation_end}'
    )

    main(train_end, validation_end)
