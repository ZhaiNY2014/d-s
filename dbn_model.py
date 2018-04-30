#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import dbn.tensorflow.models as dbn_model
import dbn.utils as utils


class DBN(object):
    """
    step1: 数据和分类分离
    step2: 初始化dbn模块
    step3: 训练dbn
    step4: 载入bp网络
    step5: 返回5维的数据及其分类
    """
    def __init__(self, dataset):
        self._input_data = dataset

    def do_dbn(self):
        # Loading dataset
        digits = self._input_data[0]
        X, Y = digits[0], digits[1]

        # Training
        classifier = dbn_model.SupervisedDBNClassification(hidden_layers_structure=[100, 80, 50, 25, 5],
                                                 learning_rate_rbm=0.05,
                                                 learning_rate=0.1,
                                                 n_epochs_rbm=20,
                                                 n_iter_backprop=100,
                                                 batch_size=100,
                                                 activation_function='sigmoid',
                                                 dropout_p=0.2)
        classifier.fit(np.array(X))

        # Test
        Y_pred = classifier.predict(X[0])

    def do_dbn_with_weight_matrix(self, path):
        dbn_model.compute_low_dimensions_data_matrix(
            utils.load_weight_matrix(path), self._input_data)








