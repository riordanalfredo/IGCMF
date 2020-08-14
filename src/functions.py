import numpy as np
import scipy.sparse as sp
import torch

def matrix_labels(M, testing=False, rating_map=None, post_rating_map=None):
    num_users = M.shape[0]
    num_items = M.shape[1]

    u_nodes_ratings = np.where(M)[0]
    v_nodes_ratings = np.where(M)[1]

    ratings = M[np.where(M)]

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(
        np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(
        np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])


    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number training
    num_train = np.where(Otraining)[0].shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    # non_zero arrays
    pairs_nonzero_train = np.array([[u, v] for u, v in zip(
        np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array(
        [u * num_items + v for u, v in pairs_nonzero_train])

     # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate(
        [pairs_nonzero_train, pairs_nonzero_test], axis=0)

    train_idx = idx_nonzero[num_val:num_train + num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    train_labels = labels[train_idx]

    class_values = np.sort(np.unique(ratings))

    '''Note here rating matrix elements' values + 1 !!!'''

    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)

    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array(
            [post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.


    rating_mx_train = sp.csr_matrix(
        rating_mx_train.reshape(num_users, num_items))

    if u_features is not None:
        u_features = sp.csr_matrix(u_features)
        print("User features shape: " + str(u_features.shape))

    if v_features is not None:
        v_features = sp.csr_matrix(v_features)
        print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, class_values