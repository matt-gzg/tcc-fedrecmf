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
        logging.FileHandler('grid_search_centralized.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

# ── Ratios (mantidos fixos) ────────────────────────────────────────────────────
train_ratio      = 0.1
validation_ratio = 0.1
k                = 20

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID = {
    'hidden_dim': [16, 32, 64],
    'lr':         [0.001, 0.005, 0.01],
    'reg':        [0.001, 0.01, 0.1],
    'n_iter':     [5, 10, 20],
}

CSV_PATH = 'grid_results_centralized.csv'
CSV_FIELDS = ['hidden_dim', 'lr', 'reg', 'n_iter', 'hr_at_k', 'total_time_s']

# ── Atualização SGD ────────────────────────────────────────────────────────────
def user_item_update(user_vector, item_vector, lr, reg_u, reg_v):
    error     = 1.0 - np.dot(user_vector, item_vector)
    grad_user = -2 * error * item_vector + 2 * reg_u * user_vector
    grad_item = -2 * error * user_vector + 2 * reg_v * item_vector
    return user_vector - lr * grad_user, item_vector - lr * grad_item

# ── Treino + avaliação para uma combinação de hiperparâmetros ──────────────────
def run(hidden_dim, lr, reg, n_iter, train_end, validation_end):
    np.random.seed(42)
    t0 = time.perf_counter()

    users_matrix = 0.1 * np.random.randn(len(user_id_list), hidden_dim)
    items_matrix = 0.1 * np.random.randn(len(item_id_list), hidden_dim)

    user_id_map      = {uid: i for i, uid in enumerate(user_id_list)}
    seen_items_online = {uid: set() for uid in user_id_list}
    hits = 0
    total_predictions = 0

    # Fase 1 — batch
    train_data = list(stream_data[:train_end])
    for epoch in range(n_iter):
        np.random.shuffle(train_data)
        for uid, item_id in train_data:
            ui = user_id_map[uid]
            users_matrix[ui], items_matrix[item_id] = user_item_update(
                users_matrix[ui], items_matrix[item_id], lr, reg, reg
            )

    for uid, item_id in train_data:
        seen_items_online[uid].add(item_id)

    # Fase 2 — prequencial
    items_matrix_T = items_matrix.T.copy()

    for uid, item_id in stream_data[train_end:validation_end]:
        ui = user_id_map[uid]

        p = np.dot(users_matrix[ui], items_matrix_T)
        seen = seen_items_online[uid]
        if seen:
            p[list(seen)] = -np.inf

        top_k = np.argpartition(-p, k)[:k]
        if item_id in top_k:
            hits += 1
        total_predictions += 1

        users_matrix[ui], items_matrix[item_id] = user_item_update(
            users_matrix[ui], items_matrix[item_id], lr, reg, reg
        )
        items_matrix_T[:, item_id] = items_matrix[item_id]
        seen_items_online[uid].add(item_id)

    hr    = hits / total_predictions if total_predictions > 0 else 0.0
    elapsed = time.perf_counter() - t0
    return hr, elapsed

# ── Grid search ────────────────────────────────────────────────────────────────
def main():
    n              = len(stream_data)
    train_end      = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    logger.info(f'Grid search | {total} combinações | Dataset={n} | Train={train_end} | Val={validation_end - train_end}')

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for idx, combo in enumerate(combos, start=1):
            params = dict(zip(keys, combo))
            logger.info(f'[{idx}/{total}] Iniciando: {params}')

            hr, elapsed = run(
                hidden_dim = params['hidden_dim'],
                lr         = params['lr'],
                reg        = params['reg'],
                n_iter     = params['n_iter'],
                train_end      = train_end,
                validation_end = validation_end,
            )

            row = {**params, 'hr_at_k': round(hr, 6), 'total_time_s': round(elapsed, 2)}
            writer.writerow(row)
            f.flush()

            logger.info(f'[{idx}/{total}] HR@{k}={hr:.4f} | Time={elapsed:.1f}s | {params}')

    logger.info(f'Grid search concluído. Resultados em {CSV_PATH}')

if __name__ == '__main__':
    main()