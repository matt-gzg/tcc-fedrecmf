import sys
import time
import numpy as np

from shared_parameter import *
from load_data import train_data, test_data, user_id_list, item_id_list, ratings_dict

#só foi tirar o encript e tals
def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = {}
    for item_id, rate, _ in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = lr * (-2 * error * single_user_vector + 2 * reg_v * item_vector[item_id])
    return single_user_vector, gradient

#essa é identica ao part
def loss():
    loss = []
    for i in range(len(user_id_list)):
        for r in range(len(train_data[user_id_list[i]])):
            item_id, rate, _ = train_data[user_id_list[i]][r]
            error = (rate - np.dot(user_vector[i], item_vector[item_id])) ** 2
            loss.append(error)
    return np.mean(loss)

#tirei a parte de cache e tempo de transmissao pq sim
if __name__ == '__main__':
    # Init process (caba tava na preguiça)
    user_vector = np.zeros([len(user_id_list), hidden_dim]) + 0.01
    item_vector = np.zeros([len(item_id_list), hidden_dim]) + 0.01

    #treino
    for iteration in range(max_iteration):
        print('###################')
        print('Iteration', iteration)

        # Step 2 User updates

        gradient_from_user = []
        user_time_list = []
        for i in range(len(user_id_list)):
            t = time.time()
            user_vector[i], gradient = user_update(user_vector[i], train_data[user_id_list[i]], item_vector)
            user_time_list.append(time.time() - t)
            print('User-%s update using' % i, user_time_list[-1], 'seconds')
            gradient_from_user.append(gradient)
        print('User Average time', np.mean(user_time_list))

        # Step 3 Server update
        t = time.time()
        for g in gradient_from_user:
            for item_id in g:
                #indo contra o gradiente
                item_vector[item_id] -= g[item_id]
        server_update_time = time.time() - t
        print('Server update using', server_update_time, 'seconds')

        # for computing loss
        print('loss', loss())
        print('Costing', max(user_time_list) + server_update_time, 'seconds')

    prediction = []
    real_label = []

    # testing

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