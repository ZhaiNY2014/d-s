import numpy as np
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
    package_path = "D:/PycharmProjects/DBN-SVM/save/weight_matrix"
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


def load_weight_matrix(path):
    weight_array_list = list()

    with open(path + "/122to100.txt") as rbm_weight_0:
        list_122to100 = list()
        for _line in rbm_weight_0.readlines():
            line = list()
            for _value in _line.strip().split(' '):
                value = float(_value.strip())
                line.append(value)
            list_122to100.append(line)
        array_0 = np.array(list_122to100)
        weight_array_list.append(array_0)

    with open(path + "/100to80.txt") as rbm_weight_1:
        list_100to80 = list()
        for _line in rbm_weight_1.readlines():
            line = list()
            for _value in _line.strip().split(' '):
                value = float(_value.strip())
                line.append(value)
            list_100to80.append(line)
        array_1 = np.array(list_100to80)
        weight_array_list.append(array_1)

    with open(path + "/80to50.txt") as rbm_weight_2:
        list_80to50 = list()
        for _line in rbm_weight_2.readlines():
            line = list()
            for _value in _line.strip().split(' '):
                value = float(_value.strip())
                line.append(value)
            list_80to50.append(line)
        array_2 = np.array(list_80to50)
        weight_array_list.append(array_2)

    with open(path + "/50to25.txt") as rbm_weight_3:
        list_50to25 = list()
        for _line in rbm_weight_3.readlines():
            line = list()
            for _value in _line.strip().split(' '):
                value = float(_value.strip())
                line.append(value)
            list_50to25.append(line)
        array_3 = np.array(list_50to25)
        weight_array_list.append(array_3)

    with open(path + "/25to5.txt") as rbm_weight_4:
        list_25to5 = list()
        for _line in rbm_weight_4.readlines():
            line = list()
            for _value in _line.strip().split(' '):
                value = float(_value.strip())
                line.append(value)
            list_25to5.append(line)
        array_4 = np.array(list_25to5)
        weight_array_list.append(array_4)

    return weight_array_list
