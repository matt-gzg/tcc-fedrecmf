import sys
import time
import numpy as np

from shared_parameter import *
from load_data import train_data, test_data, user_id_list, item_id_list

def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = {}
    for item_id, rate, _ in user_rating_list:
        if rate >= 4:
            p_ui = 1.0
        else:
            p_ui = 0.0

        error = p_ui - np.dot(single_user_vector, item_vector[item_id])
        gradient[item_id] = lr * (-2 * p_ui * error * single_user_vector + 2 * reg_v * item_vector[item_id])

        #definir um hiperparametro pra rodar varias vezes pra cada user
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
                    p = np.dot(user_vector[i], item_vector[next_item_id]) #nao ta sendo usado ainda

                new_interaction = [(item_id, rate, timestamp)]
                user_vector[i], gradient = user_update(user_vector[i], new_interaction, item_vector)

                for iid in gradient:
                    item_vector[iid] -= gradient[iid]

            user_time_list.append(time.time() - t)
            print('User-%s update using' % i, user_time_list[-1], 'seconds')
    
        print('User Average time', np.mean(user_time_list))
        print('loss', loss())
        print('Costing', max(user_time_list), 'seconds')

        #talvez fazer mais iteracoes por usuario antes de atualizar o server.

    prediction = []
    real_label = []

    # testing (tem que mudar toda essa parte pro precision e ndcg)

    #p é a predição de todos os itens, r é a real
    for i, uid in enumerate(user_id_list):
        r = test_data[uid]
        if len(r) == 0: #tem que ter uma avaliação (ver isso com o muriloso se ta certo)
            continue
        p = np.dot(user_vector[i:i + 1], np.transpose(item_vector))[0]
        prediction.extend([p[item_id] for item_id, _, _ in r])
        real_label.extend([rate for _, rate, _ in r])

    prediction = np.array(prediction, dtype=np.float32)
    real_label = np.array(real_label, dtype=np.float32)

    print('rmse', np.sqrt(np.mean(np.square(real_label - prediction))))
    print('Total time', time.time() - time_dataset, 'seconds')