import sys
import time
import numpy as np

from shared_parameter import *
from load_data import train_data, test_data, user_id_list, item_id_list, print_dataset_stats

def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = {}
    for item_id, rate, _ in user_rating_list:
        if rate >= 4:
            p_ui = 1.0
        else:
            p_ui = 0.0

        error = p_ui - np.dot(single_user_vector, item_vector[item_id])
        gradient[item_id] = lr * (-2 * p_ui * error * single_user_vector + 2 * reg_v * item_vector[item_id])

        for _ in range(hiperparam):
            single_user_vector = single_user_vector - lr * (-2 * p_ui * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
            gradient[item_id] = lr * (-2 * p_ui * error * single_user_vector + 2 * reg_v * item_vector[item_id])

    return single_user_vector, gradient

#identica ao part
def loss():
    errors = []
    for i in range(len(user_id_list)):
        for item_id, rate, _ in train_data[user_id_list[i]]:
            if rate >= 4:
                p_ui = 1.0
            else:
                p_ui = 0.0
            error = (p_ui - np.dot(user_vector[i], item_vector[item_id])) ** 2
            errors.append(error)
    return np.mean(np.array(errors, dtype=np.float128))

#tirei a parte de cache e tempo de transmissao pq sim
if __name__ == '__main__':
    time_dataset = time.time()

    print_dataset_stats()

    # Init process (caba tava na preguiça)
    user_vector = np.zeros([len(user_id_list), hidden_dim]) + 0.01
    item_vector = np.zeros([len(item_id_list), hidden_dim]) + 0.01

    #treino
    for iteration in range(max_iteration):
        print('###################')
        print('Iteration', iteration)

        # Step 2 User updates
        user_time_list = []
        for i in range(len(user_id_list)):
            t = time.time()

            uid = user_id_list[i]
            interactions = train_data[uid]

            for j, (item_id, rate, timestamp) in enumerate(interactions):
                if j < len(interactions) - 1:
                    next_item_id = interactions[j + 1][0]
                    
                    interacted = set(iid for iid, _, _ in interactions)
                    neg_candidates = [iid for iid in range(len(item_id_list)) if iid not in interacted]
                    neg_samples = np.random.choice(neg_candidates, size=min(num_neg_samples, len(neg_candidates)), replace=False)
                    
                    # Positivo: próximo item real (p_ui = 1)
                    pos_error = 1.0 - np.dot(user_vector[i], item_vector[next_item_id])
                    user_vector[i] += lr * 2 * pos_error * item_vector[next_item_id]
                    item_vector[next_item_id] += lr * 2 * pos_error * user_vector[i]
                    
                    # Negativos: itens amostrados (p_ui = 0)
                    for neg_id in neg_samples:
                        neg_error = 0.0 - np.dot(user_vector[i], item_vector[neg_id])
                        user_vector[i] += lr * 2 * neg_error * item_vector[neg_id]
                        item_vector[neg_id] += lr * 2 * neg_error * user_vector[i]

            user_time_list.append(time.time() - t)
            print('User-%s update using' % i, user_time_list[-1], 'seconds')
    
        print('User Average time', np.mean(user_time_list))
        print('loss', loss())
        print('Costing', max(user_time_list), 'seconds')

        # Evaluation after each iteration
        precision_list = []
        ndcg_list = []

        for i, uid in enumerate(user_id_list):
            test_items = test_data[uid]
            if len(test_items) == 0:
                continue
            
            # Get relevant items (rating >= 4)
            relevant_items = set(item_id for item_id, rate, _ in test_items if rate >= 4)
            
            # Predict scores for all items
            p = np.dot(user_vector[i], item_vector.T)
            
            # Rank all items by predicted score descending
            ranked_items = np.argsort(p)[::-1][:k]  # top-k item indices
            
            # Precision@k
            num_relevant_in_top_k = sum(1 for item_id in ranked_items if item_id in relevant_items)
            precision = num_relevant_in_top_k / k
            precision_list.append(precision)
            
            # NDCG@k
            dcg = 0.0
            idcg = 0.0
            for rank in range(k):
                item_id = ranked_items[rank]
                rel = 1 if item_id in relevant_items else 0
                dcg += rel / np.log2(rank + 2)  # rank starts from 1, so rank+2
            
            # IDCG: ideal DCG with relevant items at top
            num_relevant = len(relevant_items)
            for rank in range(min(k, num_relevant)):
                idcg += 1 / np.log2(rank + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)

        print('Precision@{}: {:.4f}'.format(k, np.mean(precision_list)))
        print('NDCG@{}: {:.4f}'.format(k, np.mean(ndcg_list)))

    print('Total time', time.time() - time_dataset, 'seconds')