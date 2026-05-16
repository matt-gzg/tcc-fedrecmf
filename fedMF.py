import sys
import time
import numpy as np

from shared_parameter import *
from load_data import train_data, test_data, user_id_list, item_id_list, global_train

#faz temporal para cada user separado (como localmente por aparelho)
def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = {}
    for item_id, rate, _ in user_rating_list:
        p_ui = 1.0 if rate >= 4 else 0.0

        v = item_vector[item_id]

        for _ in range(hiperparam):
            error = p_ui - np.dot(single_user_vector, v)
            single_user_vector = single_user_vector - lr * (-2 * p_ui * error * v + 2 * reg_u * single_user_vector)
            gradient[item_id] = lr * (-2 * p_ui * error * single_user_vector + 2 * reg_v * v)

        error = p_ui - np.dot(single_user_vector, v)   
        gradient[item_id] = lr * (-2 * p_ui * error * single_user_vector + 2 * reg_v * v)

    return single_user_vector, gradient

def loss(item_vector):
    total_error = 0.0
    count = 0

    for i, uid in enumerate(user_id_list):
        u = user_vector[i]
        for item_id, rate, _ in train_data[uid]:
            p_ui = 1.0 if rate >= 4 else 0.0
            error = p_ui - np.dot(u, item_vector[item_id])
            total_error += error ** 2
            count += 1

    return total_error / count

#apenas copiando por enquanto (e talvez fique assim)
def aggregate_fedavg(item_vector_global, item_vector_local, updated_items):
    for item_id in updated_items:
        item_vector_global[item_id] = item_vector_local[item_id].copy()
    item_vector_local = item_vector_global.copy()
    return item_vector_global, item_vector_local

if __name__ == '__main__':
    time_dataset = time.time()

    #mapea userid pra indice
    user_id_map = {uid: i for i, uid in enumerate(user_id_list)}
    
    #pra agregacao 
    rating_count = 0

    #init server (agora com alguma coisa decente)
    user_vector = 0.01 * np.random.randn(len(user_id_list), hidden_dim)
    
    #itens dos dispositivos e do server
    item_vector_global = 0.01 * np.random.randn(len(item_id_list), hidden_dim)
    item_vector_local  = item_vector_global.copy()

    def evaluate(user_idx, uid, data, item_vector):
        items = data[uid]
        if not items:
            return None, None
        
        relevant_items = set(item_id for item_id, rate, _ in items if rate >= 4)
        if not relevant_items:
            return None, None
        
        p = np.dot(user_vector[user_idx], item_vector.T)
        ranked_items =  np.argsort(p)[::-1][:k]
        num_relevant = len(relevant_items)

        dcg, idcg, hits = 0.0, 0.0, 0
        for rank in range(k):
            rel = 1 if ranked_items[rank] in relevant_items else 0
            hits += rel
            dcg += rel / np.log2(rank + 2)
            if rank < num_relevant:
                idcg += 1 / np.log2(rank + 2)

        precision = hits / k
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return precision, ndcg

    #treino prequencial
    # Step 2 User updates
    precision_list, ndcg_list = [], []
    
    user_time_list = []
    updated_items = set()

    last_aggregation_time = time.time()
    for uid, item_id, rate, timestamp in global_train:
        t = time.time()
        user_idx = user_id_map[uid]
        
        p, n = evaluate(user_idx, uid, test_data, item_vector_local)
        if p is not None:
            precision_list.append(p)
            ndcg_list.append(n)

        single_rating = [(item_id, rate, timestamp)]
        user_vector[user_idx], gradient = user_update(user_vector[user_idx], single_rating, item_vector_local)

        for iid, grad in gradient.items():
            item_vector_local[iid] -= grad
            updated_items.add(iid)

        rating_count += 1

        if rating_count % aggregation_int == 0:
            item_vector_global, item_vector_local = aggregate_fedavg(item_vector_global, item_vector_local, updated_items)
            updated_items = set()
            aggregation_time = time.time() - last_aggregation_time
            print(f'[Agregação #{rating_count // aggregation_int}] ratings={rating_count} | loss={loss(item_vector_local):.6f} | tempo={aggregation_time:.2f}s')
            last_aggregation_time = time.time()

        user_time_list.append(time.time() - t)
        print('User-%s update using' % uid, user_time_list[-1], 'seconds')

    print('User Average time', np.mean(user_time_list))
    print('Prequential Precision@{}: {:.4f}'.format(k, np.mean(precision_list)))
    print('Prequential NDCG@{}: {:.4f}'.format(k, np.mean(ndcg_list)))
    print('loss', loss(item_vector_local))
    print('Costing', max(user_time_list), 'seconds')

    final_precision_list, final_ndcg_list = [], []

    for i, uid in enumerate(user_id_list):
        p, n = evaluate(i, uid, test_data, item_vector_global)
        if p is not None:
            final_precision_list.append(p)
            final_ndcg_list.append(n)

    print('Precision@{}: {:.4f}'.format(k, np.mean(final_precision_list)))
    print('NDCG@{}: {:.4f}'.format(k, np.mean(final_ndcg_list)))
    print('Total time', time.time() - time_dataset, 'seconds')