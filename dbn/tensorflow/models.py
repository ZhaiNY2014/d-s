#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import atexit
from abc import ABCMeta

import numpy as np
import tensorflow as tf

from ..models import AbstractSupervisedDBN as BaseAbstractSupervisedDBN
from ..models import BaseModel
from ..models import BinaryRBM as BaseBinaryRBM
from ..models import UnsupervisedDBN as BaseUnsupervisedDBN
from ..utils import batch_generator, to_categorical, save_weight_matrix

import time
import os
import logging


def close_session():
    sess.close()


test_dir=os.path.dirname(__file__)
root_dir=os.path.join(test_dir, '..')
root = root_dir[0:root_dir.index('DBN-SVM')+7]

sess = tf.Session()
atexit.register(close_session)

tf.logging.set_verbosity(tf.logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
log_time = time.strftime('%Y_%m_%d', time.localtime(time.time()))
handler = logging.FileHandler(root + "/log/std_log_" + log_time + '.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def compute_low_dimensions_data_matrix(weight, datas):
    _datas = datas
    weight_matrix = np.matmul(
        np.matmul(
            np.matmul(
                np.matmul(np.transpose(weight[0]), np.transpose(weight[1])
                          ), np.transpose(weight[2])
            ), np.transpose(weight[3])
        ), np.transpose(weight[4])
    )
    output_train_data = np.matmul(_datas[0][0], weight_matrix)
    output_test_data = np.matmul(_datas[1][0], weight_matrix)

    save_dbn_weight_as_svm((output_train_data, _datas[0][1]), 'train')
    save_dbn_weight_as_svm((output_test_data, _datas[1][1]), 'test')


def save_dbn_weight_as_svm(data_set, t):
    if t == 'train':
        with open(root + '/file/data_dbn_train.txt', 'w') as train_data_file:
            for i1 in range(len(data_set[0])):
                train_label = data_set[1][i1]  # float
                train_data = data_set[0][i1].tolist()  # 5维list
                s1 = str(train_label) + ' '
                for j1 in range(len(train_data)):
                    s1 += str(j1)
                    s1 += ':'
                    s1 += str(train_data[j1]).strip()
                    s1 += ' '
                s1 += '\n'
                train_data_file.write(s1)
    if t == 'test':
        with open(root + '/file/data_dbn_test.txt', 'w') as test_data_file:
            for i2 in range(len(data_set[0])):
                test_label = data_set[1][i2]  # float
                test_data = data_set[0][i2].tolist()  # 5维list
                s2 = str(test_label) + ' '
                for j2 in range(len(test_data)):
                    s2 += str(j2)
                    s2 += ':'
                    s2 += str(test_data[j2]).strip()
                    s2 += ' '
                s2 += '\n'
                test_data_file.write(s2)


def weight_variable(func, shape, stddev, dtype=tf.float32):
    initial = func(shape, stddev=stddev, dtype=dtype)
    return tf.Variable(initial)


def bias_variable(value, shape, dtype=tf.float32):
    initial = tf.constant(value, shape=shape, dtype=dtype)
    return tf.Variable(initial)


class BaseTensorFlowModel(BaseModel):
    def save(self, save_path):
        import pickle

        with open(save_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @classmethod
    def load(cls, load_path):
        import pickle

        with open(load_path, 'rb') as fp:
            dct_to_load = pickle.load(fp)
            return cls.from_dict(dct_to_load)

    def to_dict(self):
        dct_to_save = {name: self.__getattribute__(name) for name in self._get_param_names()}
        dct_to_save.update(
            {name: self.__getattribute__(name).eval(sess) for name in self._get_weight_variables_names()}
        )
        return dct_to_save

    @classmethod
    def from_dict(cls, dct_to_load):
        pass

    def _build_model(self, weights=None):
        pass

    def _initialize_weight(self, weights):
        pass

    @classmethod
    def _get_weight_variables_names(cls):
        pass

    @classmethod
    def _get_param_names(cls):
        pass


class BinaryRBM(BaseBinaryRBM, BaseTensorFlowModel):
    """
    这个类实现了一个二进制的受限玻尔兹曼机
    """
    def fit(self, X):
        self.n_visible_units = X.shape[1]
        self._build_model()

        sess.run(tf.variables_initializer([self.W, self.c, self.b]))

        if self.optimization_algorithm == 'sgd':
            self._stochastic_gradient_descent(X)
            # print(sess.run(self.W))
        else:
            raise ValueError("Invalid optimization algorithm.")
        return

    @classmethod
    def _get_weight_variables_names(cls):
        return ['W', 'c', 'b']

    @classmethod
    def _get_param_names(cls):
        return ['n_hidden_units',
                'n_visible_units',
                'activation_function',
                'optimization_algorithm',
                'learning_rate',
                'n_epochs',
                'contrastive_divergence_iter',
                'batch_size',
                'verbose',
                '_activation_function_class']

    def _initialize_weights(self, weights):
        if weights:
            for attr_name, value in weights.items():
                self.__setattr__(attr_name, tf.Variable(value))
        else:
            if self.activation_function == 'sigmoid':
                stddev = 1.0 / np.sqrt(self.n_visible_units)
                self.W = weight_variable(tf.random_normal, [self.n_hidden_units, self.n_visible_units], stddev)
                self.c = weight_variable(tf.random_normal, [self.n_hidden_units], stddev)
                self.b = weight_variable(tf.random_normal, [self.n_visible_units], stddev)
                self._activation_function_class = tf.nn.sigmoid
            elif self.activation_function == 'relu':
                stddev = 0.1 / np.sqrt(self.n_visible_units)
                self.W = weight_variable(tf.truncated_normal, [self.n_hidden_units, self.n_visible_units], stddev)
                self.c = bias_variable(stddev, [self.n_hidden_units])
                self.b = bias_variable(stddev, [self.n_visible_units])
                self._activation_function_class = tf.nn.relu
            else:
                raise ValueError("Invalid activation function.")

    def _build_model(self, weights=None):
        """
        构建TensorFlow模块
        :param weights: 
        :return: 
        """
        self._initialize_weights(weights)
        self.visible_units_placeholder = tf.placeholder(tf.float32, shape=[None, self.n_visible_units])
        self.compute_hidden_units_op = self._activation_function_class(
            tf.transpose(tf.matmul(self.W, tf.transpose(self.visible_units_placeholder))) + self.c)
        self.hidden_units_placeholder = tf.placeholder(tf.float32, shape=[None, self.n_hidden_units])
        self.compute_visible_units_op = self._activation_function_class(
            tf.matmul(self.hidden_units_placeholder, self.W) + self.b)
        self.random_uniform_values = tf.Variable(tf.random_normal([self.batch_size, self.n_hidden_units]))
        sample_hidden_units_op = tf.to_float(self.random_uniform_values < self.compute_hidden_units_op)
        self.random_variables = [self.random_uniform_values]

        positive_gradient_op = tf.matmul(tf.expand_dims(sample_hidden_units_op, 2),
                                         tf.expand_dims(self.visible_units_placeholder, 1))

        sample_hidden_units_gibbs_step_op = sample_hidden_units_op
        for t in range(self.contrastive_divergence_iter):
            compute_visible_units_op = self._activation_function_class(
                tf.matmul(sample_hidden_units_gibbs_step_op, self.W) + self.b)
            compute_hidden_units_gibbs_step_op = self._activation_function_class(
                tf.transpose(tf.matmul(self.W, tf.transpose(compute_visible_units_op))) + self.c)
            random_uniform_values = tf.Variable(tf.random_uniform([self.batch_size, self.n_hidden_units]))
            sample_hidden_units_gibbs_step_op = tf.to_float(random_uniform_values < compute_hidden_units_gibbs_step_op)
            self.random_variables.append(random_uniform_values)

        negative_gradient_op = tf.matmul(tf.expand_dims(sample_hidden_units_gibbs_step_op, 2),
                                         tf.expand_dims(compute_visible_units_op, 1))

        compute_delta_W = tf.reduce_mean(positive_gradient_op - negative_gradient_op, 0)
        compute_delta_b = tf.reduce_mean(self.visible_units_placeholder - compute_visible_units_op, 0)
        compute_delta_c = tf.reduce_mean(sample_hidden_units_op - sample_hidden_units_gibbs_step_op, 0)

        self.update_W = tf.assign_add(self.W, self.learning_rate * compute_delta_W)
        self.update_b = tf.assign_add(self.b, self.learning_rate * compute_delta_b)
        self.update_c = tf.assign_add(self.c, self.learning_rate * compute_delta_c)

    @classmethod
    def from_dict(cls, dct_to_load):
        weights = {var_name: dct_to_load.pop(var_name) for var_name in cls._get_weight_variables_names()}
        _activation_function_class = dct_to_load.pop('_activation_function_class')
        n_visible_units = dct_to_load.pop('n_visible_units')

        instance = cls(**dct_to_load)
        setattr(instance, _activation_function_class, _activation_function_class)
        setattr(instance, 'n_visible_units', n_visible_units)

        instance._build_model(weights)
        sess.run(tf.variables_initializer([getattr(instance, name) for name in cls._get_weight_variables_names()]))
        return instance

    def _stochastic_gradient_descent(self, _data):
        """
        执行随机梯度下降优化算法
        :param _data:
        :return:
        """
        for iteration in range(1, self.n_epochs + 1):
            idx = np.random.permutation(len(_data))
            data = _data[idx]
            for batch in batch_generator(self.batch_size, data):
                if len(batch) < self.batch_size:
                    # Pad with zeros
                    pad = np.zeros((self.batch_size - batch.shape[0], batch.shape[1]), dtype=batch.dtype)
                    batch = np.vstack((batch, pad))
                sess.run(tf.variables_initializer(self.random_variables))  # Need to re-sample from uniform distribution
                sess.run([self.update_W, self.update_b, self.update_c],
                         feed_dict={self.visible_units_placeholder: batch})

            # saver = tf.train.Saver()
            # # save(saver, sess)
            # saver.save(sess, "../../save/model")
            if self.verbose:
                error = self._compute_reconstruction_error(data)
                print(">> Epoch %d finished \tRBM Reconstruction error %f , %s" %
                      (iteration, error, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                logger.warning(">> Epoch %d finished \tRBM Reconstruction error %f , %s" %
                      (iteration, error, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        self.W_list.append(sess.run(self.W))
        rbm_hidden_size = self.W.get_shape()._dims[0].value
        rbm_visible_size = self.W.get_shape()._dims[1].value
        save_weight_matrix(sess.run(self.W), rbm_visible_size, rbm_hidden_size)

    def _compute_hidden_units_matrix(self, matrix_visible_units):
        """
        计算隐层单元的输出
        :param matrix_visible_units: 
        :return: 
        """
        return sess.run(self.compute_hidden_units_op,
                        feed_dict={self.visible_units_placeholder: matrix_visible_units})

    def _compute_visible_units_matrix(self, matrix_hidden_units):
        """
        计算显层单元的输出
        :param matrix_hidden_units: 
        :return: 
        """
        return sess.run(self.compute_visible_units_op,
                        feed_dict={self.hidden_units_placeholder: matrix_hidden_units})


class UnsupervisedDBN(BaseUnsupervisedDBN, BaseTensorFlowModel):
    """
    无监督的基于TensorFlow实现的DBN
    """
    def __init__(self, **kwargs):
        super(UnsupervisedDBN, self).__init__(**kwargs)
        self.rbm_class = BinaryRBM

    @classmethod
    def _get_param_names(cls):
        return ['hidden_layers_structure',
                'activation_function',
                'optimization_algorithm',
                'learning_rate_rbm',
                'n_epochs_rbm',
                'contrastive_divergence_iter',
                'batch_size',
                'verbose']

    @classmethod
    def _get_weight_variables_names(cls):
        return []

    def to_dict(self):
        dct_to_save = super(UnsupervisedDBN, self).to_dict()
        dct_to_save['rbm_layers'] = [rbm.to_dict() for rbm in self.rbm_layers]
        return dct_to_save

    @classmethod
    def from_dict(cls, dct_to_load):
        rbm_layers = dct_to_load.pop('rbm_layers')
        instance = cls(**dct_to_load)
        setattr(instance, 'rbm_layers', [instance.rbm_class.from_dict(rbm) for rbm in rbm_layers])
        return instance


class TensorFlowAbstractSupervisedDBN(BaseAbstractSupervisedDBN, BaseTensorFlowModel):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        super(TensorFlowAbstractSupervisedDBN, self).__init__(UnsupervisedDBN, **kwargs)

    @classmethod
    def _get_param_names(cls):
        return ['n_iter_backprop',
                'l2_regularization',
                'learning_rate',
                'batch_size',
                'dropout_p',
                'verbose']

    @classmethod
    def _get_weight_variables_names(cls):
        return ['W', 'b']

    def _initialize_weights(self, weights):
        if weights:
            for attr_name, value in weights.items():
                self.__setattr__(attr_name, tf.Variable(value))
        else:
            if self.unsupervised_dbn.activation_function == 'sigmoid':
                stddev = 1.0 / np.sqrt(self.input_units)
                self.W = weight_variable(tf.random_normal, [self.input_units, self.num_classes], stddev)
                self.b = weight_variable(tf.random_normal, [self.num_classes], stddev)
                self._activation_function_class = tf.nn.sigmoid
            elif self.unsupervised_dbn.activation_function == 'relu':
                stddev = 0.1 / np.sqrt(self.input_units)
                self.W = weight_variable(tf.truncated_normal, [self.input_units, self.num_classes], stddev)
                self.b = bias_variable(stddev, [self.num_classes])
                self._activation_function_class = tf.nn.relu
            else:
                raise ValueError("Invalid activation function.")

    def to_dict(self):
        dct_to_save = super(TensorFlowAbstractSupervisedDBN, self).to_dict()
        dct_to_save['unsupervised_dbn'] = self.unsupervised_dbn.to_dict()
        dct_to_save['num_classes'] = self.num_classes
        return dct_to_save

    @classmethod
    def from_dict(cls, dct_to_load):
        weights = {var_name: dct_to_load.pop(var_name) for var_name in cls._get_weight_variables_names()}
        unsupervised_dbn_dct = dct_to_load.pop('unsupervised_dbn')
        num_classes = dct_to_load.pop('num_classes')

        instance = cls(**dct_to_load)

        setattr(instance, 'unsupervised_dbn', instance.unsupervised_dbn_class.from_dict(unsupervised_dbn_dct))
        setattr(instance, 'num_classes', num_classes)

        instance._build_model(weights)
        sess.run(tf.variables_initializer([getattr(instance, name) for name in cls._get_weight_variables_names()]))
        return instance

    def _build_model(self, weights=None):
        self.visible_units_placeholder = self.unsupervised_dbn.rbm_layers[0].visible_units_placeholder
        keep_prob = tf.placeholder(tf.float32)
        visible_units_placeholder_drop = tf.nn.dropout(self.visible_units_placeholder, keep_prob)
        self.keep_prob_placeholders = [keep_prob]

        rbm_activation = visible_units_placeholder_drop
        for rbm in self.unsupervised_dbn.rbm_layers:
            rbm_activation = rbm._activation_function_class(
                tf.transpose(tf.matmul(rbm.W, tf.transpose(rbm_activation))) + rbm.c)
            keep_prob = tf.placeholder(tf.float32)
            self.keep_prob_placeholders.append(keep_prob)
            rbm_activation = tf.nn.dropout(rbm_activation, keep_prob)

        self.transform_op = rbm_activation
        self.input_units = self.unsupervised_dbn.rbm_layers[-1].n_hidden_units
        self._initialize_weights(weights)

        if self.unsupervised_dbn.optimization_algorithm == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise ValueError("Invalid optimization algorithm.")

        self.y = tf.matmul(self.transform_op, self.W) + self.b
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.train_step = None
        self.cost_function = None
        self.output = None

    def _fine_tuning(self, data, _labels):
        self.num_classes = self._determine_num_output_neurons(_labels)
        if self.num_classes == 1:
            _labels = np.expand_dims(_labels, -1)

        self._build_model()
        sess.run(tf.variables_initializer([self.W, self.b]))

        labels = self._transform_labels_to_network_format(_labels)

        if self.verbose:
            print("[START] Fine tuning step:")
        self._stochastic_gradient_descent(data, labels)
        if self.verbose:
            print("[END] Fine tuning step")

    def _stochastic_gradient_descent(self, data, labels):
        for iteration in range(self.n_iter_backprop):
            for batch_data, batch_labels in batch_generator(self.batch_size, data, labels):
                feed_dict = {self.visible_units_placeholder: batch_data,
                             self.y_: batch_labels}
                feed_dict.update({placeholder: self.p for placeholder in self.keep_prob_placeholders})
                sess.run(self.train_step, feed_dict=feed_dict)

            if self.verbose:
                feed_dict = {self.visible_units_placeholder: data, self.y_: labels}
                feed_dict.update({placeholder: 1.0 for placeholder in self.keep_prob_placeholders})
                error = sess.run(self.cost_function, feed_dict=feed_dict)
                print(">> Epoch %d finished \tANN training loss %f" % (iteration, error))

    def transform(self, X):
        feed_dict = {self.visible_units_placeholder: X}
        feed_dict.update({placeholder: 1.0 for placeholder in self.keep_prob_placeholders})
        return sess.run(self.transform_op,
                        feed_dict=feed_dict)

    def predict(self, X):
        if len(X.shape) == 1:  # It is a single sample
            X = np.expand_dims(X, 0)
        predicted_data = self._compute_output_units_matrix(X)
        return predicted_data

    def _compute_output_units_matrix(self, matrix_visible_units):
        feed_dict = {self.visible_units_placeholder: matrix_visible_units}
        feed_dict.update({placeholder: 1.0 for placeholder in self.keep_prob_placeholders})
        return sess.run(self.output, feed_dict=feed_dict)


class SupervisedDBNClassification(TensorFlowAbstractSupervisedDBN):
    def _build_model(self, weights=None):
        super(SupervisedDBNClassification, self)._build_model(weights)
        self.output = tf.nn.softmax(self.y)
        self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
        self.train_step = self.optimizer.minimize(self.cost_function)

    @classmethod
    def _get_param_names(cls):
        return super(SupervisedDBNClassification, cls)._get_param_names() + ['label_to_idx_map', 'idx_to_label_map']

    @classmethod
    def from_dict(cls, dct_to_load):
        label_to_idx_map = dct_to_load.pop('label_to_idx_map')
        idx_to_label_map = dct_to_load.pop('idx_to_label_map')

        instance = super(SupervisedDBNClassification, cls).from_dict(dct_to_load)
        setattr(instance, 'label_to_idx_map', label_to_idx_map)
        setattr(instance, 'idx_to_label_map', idx_to_label_map)

        return instance

    def _transform_labels_to_network_format(self, labels):
        new_labels, label_to_idx_map, idx_to_label_map = to_categorical(labels, self.num_classes)
        self.label_to_idx_map = label_to_idx_map
        self.idx_to_label_map = idx_to_label_map
        return new_labels

    def _transform_network_format_to_labels(self, indexes):
        return list(map(lambda idx: self.idx_to_label_map[idx], indexes))

    def predict(self, X):
        probs = self.predict_proba(X)
        indexes = np.argmax(probs, axis=1)
        return self._transform_network_format_to_labels(indexes)

    def predict_proba(self, X):
        return super(SupervisedDBNClassification, self)._compute_output_units_matrix(X)

    def predict_proba_dict(self, X):
        if len(X.shape) == 1:  # It is a single sample
            X = np.expand_dims(X, 0)

        predicted_probs = self.predict_proba(X)

        result = []
        num_of_data, num_of_labels = predicted_probs.shape
        for i in range(num_of_data):
            dict_prob = {}
            for j in range(num_of_labels):
                dict_prob[self.idx_to_label_map[j]] = predicted_probs[i][j]
            result.append(dict_prob)

        return result

    def _determine_num_output_neurons(self, labels):
        return len(np.unique(labels))

