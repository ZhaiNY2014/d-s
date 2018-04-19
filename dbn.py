#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
np.random.seed(1337)  # for reproducibility


class DBN(object):
    """
    step1: 数据和分类分离
    step2: 初始化dbn模块
    step3: 训练dbn
    step4: 载入bp网络
    step5: 返回5维的数据及其分类
    
    """
    def __init__(self, dataset):
        pass

    # Loading dataset
    digits = load_digits()
    X, Y = digits.data, digits.target

    # Data scaling
    X = (X / 16).astype(np.float32)

    # Splitting data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Training
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.fit(X_train, Y_train)

    # Test
    Y_pred = classifier.predict(X_test)
    print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))


