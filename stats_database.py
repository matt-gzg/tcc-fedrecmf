from load_data import ratings_dict, user_id_list, item_id_list, train_data, test_data

num_users = len(user_id_list)
num_items = len(item_id_list)
num_ratings = sum(len(user_ratings) for user_ratings in ratings_dict.values())
total = num_users * num_items
sparsity = 1.0 - (num_ratings / total)

print('Number of users:', num_users)
print('Number of items:', num_items)
print('Number of ratings:', num_ratings)
print('Sparsity: {:.6f}'.format(sparsity))
print('Number of training ratings:', sum(len(train_data[u]) for u in train_data))
print('Number of testing  ratings:', sum(len(test_data[u])  for u in test_data))
