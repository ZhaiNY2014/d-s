#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math


def doDBN(data):
    _data = data
    opts = DLOption(10, 1., 100, 0.0, 0., 0.)
    dbn = DBN([100, 80, 50, 25, 5], opts, _data)
    dbn.train()
    return _data


class DBN:
    def __init__(self, sizes, opts, data):
        self._sizes = sizes
        self._opts = opts
        self._data = data
        self.rbm_list = []
        input_size = data.shape[1]
        for i, size in enumerate(self._sizes):
            self.rbm_list.append(RBM("rbm%d" % i), input_size, size, self._opts)
            input_size = size

    def train(self):
        data = self._data
        for rbm in self.rbm_list:
            rbm.train(data)
            data = rbm.rbmup(data)


class DLOption(object):

    def __init__(self, epoches, learning_rate, batchsize, momentum, penaltyL2, dropoutProb):
        '''
        :param epoches: 1个epoch等于使用训练集中的全部样本训练一次
        :param learning_rate: 学习率
        :param batchsize: 批大小
        :param momentum:  动量 
        :param penaltyL2:  L2范数
        :param dropoutProb: 
        '''
        self._epoches = epoches
        self._learning_rate = learning_rate
        self._batchsize = batchsize
        self._momentum = momentum
        self._penaltyL2 = penaltyL2
        self._dropoutProb = dropoutProb


class RBM(object):
    def __init__(self, name, input_size, output_size, opts):
        self._name = name
        self._input_size = input_size
        self._output_size = output_size
        self._opts = opts
        self.init_w = np.zeros([input_size, output_size], np.float32)
        self.init_hb = np.zeros([output_size], np.float32)
        self.init_vb = np.zeros([input_size], np.float32)
        self.w = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)

    def reset_init_parameter(self, init_weights, init_hbias, init_vbias):
        self.init_w = init_weights
        self.init_hb = init_hbias
        self.init_vb = init_vbias

    def propup(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def propdown(self, hidden, w, vb):
        return tf.nn.sigmoid(
            tf.matmul(hidden, tf.transpose(w)) + vb)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def rbmup(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run(out)

    def train(self, X):
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        _vw = tf.placeholder("float", [self._input_size, self._output_size])
        _vhb = tf.placeholder("float", [self._output_size])
        _vvb = tf.placeholder("float", [self._input_size])
        _current_vw = np.zeros(
            [self._input_size, self._output_size], np.float32)
        _current_vhb = np.zeros([self._output_size], np.float32)
        _current_vvb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])
        h0 = self.sample_prob(self.propup(v0, _w, _hb))
        v1 = self.sample_prob(self.propdown(h0, _w, _vb))
        h1 = self.propup(v1, _w, _hb)
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        update_vw = _vw * self._opts._momentum + self._opts._learning_rate * \
            (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vvb = _vvb * self._opts._momentum + \
            self._opts._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_vhb = _vhb * self._opts._momentum + \
            self._opts._learning_rate * tf.reduce_mean(h0 - h1, 0)
        update_w = _w + _vw
        update_vb = _vb + _vvb
        update_hb = _hb + _vhb
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            old_w = self.init_w
            old_hb = self.init_hb
            old_vb = self.init_vb
            # TODO(train): rbm的具体训练内容

            for i in range(self._opts._epoches):
                for start, end in zip(range(0, len(X), self._opts._batchsize),
                                      range(self._opts._batchsize,
                                            len(X), self._opts._batchsize)):
                    batch = X[start:end]
                    _current_vw = sess.run(update_vw, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vw: _current_vw})
                    _current_vhb = sess.run(update_vhb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vhb: _current_vhb})
                    _current_vvb = sess.run(update_vvb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vvb: _current_vvb})
                    old_w = sess.run(update_w, feed_dict={
                                     _w: old_w, _vw: _current_vw})
                    old_hb = sess.run(update_hb, feed_dict={
                        _hb: old_hb, _vhb: _current_vhb})
                    old_vb = sess.run(update_vb, feed_dict={
                        _vb: old_vb, _vvb: _current_vvb})

                image = Image.fromarray(
                    tile_raster_images(
                        X=old_w.T,
                        img_shape=(int(math.sqrt(self._input_size)),
                                   int(math.sqrt(self._input_size))),
                        tile_shape=(int(math.sqrt(self._output_size)),
                                    int(math.sqrt(self._output_size))),
                        tile_spacing=(1, 1)
                    )
                )
                image.save("%s_%d.png" % (self._name, i))

            self.w = old_w
            self.hb = old_hb
            self.vb = old_vb


"""
class NN(object):
    def __init__(self, sizes, opts, X, Y):
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        input_size = X.shape[1]
        for size in self._sizes + [Y.shape[1]]:
            max_range = 4 * math.sqrt(6. / (input_size + size))
            self.w_list.append(
                np.random.uniform(
                    -max_range, max_range, [input_size, size]
                ).astype(np.float32))
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    def load_from_dbn(self, dbn):
        assert len(dbn._sizes) == len(self._sizes)
        for i in range(len(self._sizes)):
            assert dbn._sizes[i] == self._sizes[i]
        for i in range(len(self._sizes)):
            self.w_list[i] = dbn.rbm_list[i].w
            self.b_list[i] = dbn.rbm_list[i].hb

    def train(self):
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        cost = tf.reduce_mean(tf.square(_a[-1] - y))
        train_op = tf.train.MomentumOptimizer(
            self._opts._learning_rate, self._opts._momentum).minimize(cost)
        predict_op = tf.argmax(_a[-1], 1)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(self._opts._epoches):
                for start, end in zip(
                    range(
                        0, len(self._X),
                        self._opts._batchsize),
                    range(
                        self._opts._batchsize, len(
                            self._X),
                        self._opts._batchsize)):
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})
                for i in range(len(self._sizes) + 1):
                    self.w_list[i] = sess.run(_w[i])
                    self.b_list[i] = sess.run(_b[i])
                print(np.mean(np.argmax(self._Y, axis=1) ==
                              sess.run(predict_op, feed_dict={
                                  _a[0]: self._X, y: self._Y})))

    def predict(self, X):
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        predict_op = tf.argmax(_a[-1], 1)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run(predict_op, feed_dict={_a[0]: X})
"""
