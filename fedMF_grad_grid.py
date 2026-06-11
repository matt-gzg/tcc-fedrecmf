import time
import csv
import itertools
import numpy as np
import logging
from load_from_cache import (user_id_list, item_id_list, stream_data)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('grid_search_federated_grad.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

# ── Fixos ─────────────────────────────────────────────────────────────────────
train_ratio      = 0.1
validation_ratio = 0.1
k                = 10

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID = {
    'hidden_dim': [16, 32, 64],
    'lr':         [0.001, 0.005, 0.01],
    'reg':        [0.001, 0.01, 0.1],
    'n_iter':     [1, 5, 10],
}

CSV_PATH   = 'grid_results_federated_grad.csv'
CSV_FIELDS = ['hidden_dim', 'lr', 'reg', 'n_iter', 'aggregation_int', 'hr_at_k', 'total_time_s']

# ── Treino local — retorna gradiente puro do item ──────────────────────────────
def user_update(user_vector, item_id, v, lr, reg_u, reg_v, n_iter):
    gradient = {}
    for _ in range(n_iter):
        error     = 1.0 - np.dot(user_vector, v)
        grad_item = -2 * error * user_vector + 2 * reg_v * v
        user_vector = user_vector - lr * (-2 * error * v + 2 * reg_u * user_vector)
        gradient[item_id] = grad_item
    return user_vector, gradient

# ── Agregação FedAvg sobre gradientes ─────────────────────────────────────────
def aggregate_fedavg(items_matrix_global, items_matrix_local, interact_count,
                     updated_items, item_users_updated, item_users_all, hidden_dim, lr):
    for item_id in updated_items:
        total_interacs = 0
        weighted_sum   = np.zeros(hidden_dim)

        for uid in item_users_updated[item_id]:
            n = interact_count[uid]
            weighted_sum   += n * items_matrix_local[uid][item_id]
            total_interacs += n

        if total_interacs > 0:
            avg_gradient = weighted_sum / total_interacs
            items_matrix_global[item_id] -= lr * avg_gradient

            for uid in item_users_all[item_id]:
                items_matrix_local[uid][item_id] = items_matrix_global[item_id].copy()

    return items_matrix_global, items_matrix_local

# ── Run de uma combinação ──────────────────────────────────────────────────────
def run(hidden_dim, lr, reg, n_iter, aggregation_int, train_end, validation_end):
    np.random.seed(42)
    t0 = time.perf_counter()

    users_matrix          = 0.1 * np.random.randn(len(user_id_list), hidden_dim)
    items_matrix_global   = 0.1 * np.random.randn(len(item_id_list), hidden_dim)
    items_matrix_local    = {uid: {} for uid in user_id_list}
    items_matrix_global_T = items_matrix_global.T.copy()

    user_id_map       = {uid: i for i, uid in enumerate(user_id_list)}
    seen_items_online = {uid: set() for uid in user_id_list}

    updated_items      = set()
    interact_count     = {uid: 0 for uid in user_id_list}
    item_users_updated = {}
    item_users_all     = {}

    hits = 0
    total_predictions = 0

    for obs_count, (uid, item_id) in enumerate(stream_data[:validation_end], start=1):
        phase      = "train" if obs_count <= train_end else "validation"
        user_index = user_id_map[uid]

        if phase == 'validation':
            p = np.dot(users_matrix[user_index], items_matrix_global_T)

            if items_matrix_local[uid]:
                local_ids  = list(items_matrix_local[uid].keys())
                local_vecs = np.array(list(items_matrix_local[uid].values()))
                p[local_ids] = users_matrix[user_index] @ local_vecs.T

            seen_items = seen_items_online[uid]
            if seen_items:
                p[list(seen_items)] = -np.inf

            top_k = np.argpartition(p, -k)[-k:]
            if item_id in top_k:
                hits += 1
            total_predictions += 1

        if item_id not in items_matrix_local[uid]:
            items_matrix_local[uid][item_id] = items_matrix_global[item_id].copy()

        current_item = items_matrix_local[uid][item_id]

        users_matrix[user_index], gradient = user_update(
            users_matrix[user_index], item_id, current_item,
            lr, reg, reg, n_iter
        )
        items_matrix_local[uid][item_id] = gradient[item_id]

        updated_items.add(item_id)
        interact_count[uid] += 1

        if item_id not in item_users_updated:
            item_users_updated[item_id] = set()
        item_users_updated[item_id].add(uid)

        if item_id not in item_users_all:
            item_users_all[item_id] = set()
        item_users_all[item_id].add(uid)

        seen_items_online[uid].add(item_id)

        if obs_count % aggregation_int == 0:
            items_matrix_global, items_matrix_local = aggregate_fedavg(
                items_matrix_global, items_matrix_local, interact_count,
                updated_items, item_users_updated, item_users_all, hidden_dim, lr
            )
            items_matrix_global_T = items_matrix_global.T.copy()

            item_users_updated = {}
            updated_items      = set()
            interact_count     = {uid: 0 for uid in user_id_list}

    hr      = hits / total_predictions if total_predictions > 0 else 0.0
    elapsed = time.perf_counter() - t0
    return hr, elapsed

# ── Grid search ────────────────────────────────────────────────────────────────
def main():
    n              = len(stream_data)
    train_end      = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    aggregation_int = validation_end // 20

    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    logger.info(
        f'Grid search federado (gradientes) | {total} combinações | '
        f'Dataset={n} | Train={train_end} | Val={validation_end - train_end} | '
        f'aggregation_int={aggregation_int}'
    )

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for idx, combo in enumerate(combos, start=1):
            params = dict(zip(keys, combo))
            logger.info(f'[{idx}/{total}] Iniciando: {params}')

            hr, elapsed = run(
                hidden_dim      = params['hidden_dim'],
                lr              = params['lr'],
                reg             = params['reg'],
                n_iter          = params['n_iter'],
                aggregation_int = aggregation_int,
                train_end       = train_end,
                validation_end  = validation_end,
            )

            row = {**params, 'aggregation_int': aggregation_int, 'hr_at_k': round(hr, 6), 'total_time_s': round(elapsed, 2)}
            writer.writerow(row)
            f.flush()

            logger.info(f'[{idx}/{total}] HR@{k}={hr:.4f} | Time={elapsed:.1f}s | {params}')

    logger.info(f'Grid search concluído. Resultados em {CSV_PATH}')

if __name__ == '__main__':
    main()