#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn import datasets
from libsvm.python.svmutil import *


class SVM:
    def __init__(self):
        self._train_label, self._train_value = svm_read_problem('/file/data_dbn_train.txt')
        self._test_label, self._test_value = svm_read_problem('/file/data_dbn_test.txt')

    def train(self):
        model = svm_train(self._train_label, self._train_value)
        return model

    def predict(self, model):
        p_label, p_acc, p_val = svm_predict(self._test_label, self._test_value, model)
        return p_acc

    def do_svm(self):
        acc = SVM.predict(SVM.train(self))
        save(acc)
        print(acc)


def save(acc):
    with open("/file/data_svm.txt") as file:
        for line in acc:
            file.write(line)


def doSVM():
    # 获取数据
    # iris = self.data
    iris = datasets.load_iris()
    x_vals = np.array([[x[0], x[3]] for x in iris.data])
    y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

    # 分离训练和测试集
    train_indices = np.random.choice(len(x_vals), int(len(x_vals) * 0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    batch_size = 100

    # 初始化feedin
    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # 创建权值参数
    A = tf.Variable(tf.random_normal(shape=[2, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # 定义线性模型: y = Ax + b
    model_output = tf.subtract(tf.matmul(x_data, A), b)

    # Declare vector L2 'norm' function squared
    l2_norm = tf.reduce_sum(tf.square(A))

    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    alpha = tf.constant([0.01])
    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # 持久化
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training loop
        for i in range(20000):
            rand_index = np.random.choice(len(x_vals_train), size=batch_size)
            rand_x = x_vals_train[rand_index]
            rand_y = np.transpose([y_vals_train[rand_index]])
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        result = tf.maximum(0., tf.multiply(model_output, y_target))
        y_test = np.reshape(y_vals_test, (len(y_vals_test), 1))
        array = sess.run(result, feed_dict={x_data: x_vals_test, y_target: y_test})
        num = np.array(array)
        zero_num = np.sum(num == [0])
        print(num)
        print(zero_num)
