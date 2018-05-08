#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from libsvm.python.svmutil import *
import time
import utils


root = utils.get_root_path(False)


def do_svm():
    _train_label, _train_value = svm_read_problem(root + '/file/data_dbn_train.txt')
    _test_label, _test_value = svm_read_problem(root + '/file/data_dbn_test.txt')
    model = svm_train(_train_label, _train_value, '-s 0 -t 2')
    p_label, p_acc, p_val = svm_predict(_test_label, _test_value, model)
    save(p_acc)
    print(p_acc)


def save(acc):
    with open(root + "/file/data_svm.txt", 'a') as file:
        file.write('acc = ' + str(acc[0]) + ',' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')

