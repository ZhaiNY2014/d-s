import numpy as np
import time
import os


def batch_generator(batch_size, data, labels=None):
    n_batches = int(np.ceil(len(data) / float(batch_size)))
    idx = np.random.permutation(len(data))
    data_shuffled = data[idx]
    if labels is not None:
        labels_shuffled = labels[idx]
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if labels is not None:
            yield data_shuffled[start:end, :], labels_shuffled[start:end]
        else:
            yield data_shuffled[start:end, :]


def to_categorical(labels, num_classes):
    new_labels = np.zeros([len(labels), num_classes])
    label_to_idx_map, idx_to_label_map = dict(), dict()
    idx = 0
    for i, label in enumerate(labels):
        if label not in label_to_idx_map:
            label_to_idx_map[label] = idx
            idx_to_label_map[idx] = label
            idx += 1
        new_labels[i][label_to_idx_map[label]] = 1
    return new_labels, label_to_idx_map, idx_to_label_map


def save_weight_matrix(W, rbm_visible_size, rbm_hidden_size):
    _W = W.tolist()
    matrix4write = list()
    for _line in _W:
        line = list2str(_line)
        matrix4write.append(line)
    package_path = "D:/PycharmProjects/DBN-SVM/save/weight_matrix_" + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    if not os.path.exists(package_path):
        os.makedirs(package_path)
    file_path = '/' + str(rbm_visible_size) + 'to' + str(rbm_hidden_size) + '.txt'
    with open(package_path + file_path, 'w') as file:
        for line in matrix4write:
            file.write(line)


def list2str(_list):
    line = ""
    for X in _list:
        if isinstance(X, str):
            line = line + ' ' + X
        elif isinstance(X, list):
            sline = list2str(X)
            line = line + ' ' + sline
        elif isinstance(X, int) or isinstance(X, float):
            snum = str(X)
            line = line + ' ' + snum
    line += '\n'

    return line
