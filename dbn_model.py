#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import dbn.tensorflow.models as dbn_model
import dbn.utils as utils
import tensorflow as tf

from yadlt.models.boltzmann import dbn
from yadlt.utils import utilities


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

    def do_dbn(self, action='pp'):
        if action == 'pp':
            # Loading dataset
            digits = self._input_data[0]
            X, Y = digits[0], digits[1]

            # Training
            classifier = dbn_model.SupervisedDBNClassification(hidden_layers_structure=[100, 80, 50, 25, 5],
                                                     learning_rate_rbm=0.1,
                                                     learning_rate=0.1,
                                                     n_epochs_rbm=100,
                                                     n_iter_backprop=100,
                                                     batch_size=64,
                                                     activation_function='sigmoid',
                                                     dropout_p=0.2)
            classifier.fit(np.array(X))

            # Test
            Y_pred = classifier.predict(X[0])

        elif action == 'yadlt':
            flags = tf.app.flags
            FLAGS = flags.FLAGS

            trX, trY = np.array(self._input_data[0][0]), trans_label_to_yadlt(self._input_data[0][1])
            vlX, vlY = np.array(self._input_data[1][0][:10000]), trans_label_to_yadlt(self._input_data[1][1][:10000])
            teX, teY = np.array(self._input_data[1][0]), trans_label_to_yadlt(self._input_data[1][1])

            finetune_act_func = utilities.str2actfunc('relu')

            srbm = dbn.DeepBeliefNetwork(
                name='dbn', do_pretrain=True,
                rbm_layers=[100, 80, 50, 25, 5],
                finetune_act_func=finetune_act_func, rbm_learning_rate=[0.1],
                rbm_num_epochs=[100], rbm_gibbs_k=[1],
                rbm_gauss_visible=False, rbm_stddev=0.1,
                momentum=0.9, rbm_batch_size=[64],
                finetune_learning_rate=0.01,
                finetune_num_epochs=10, finetune_batch_size=64,
                finetune_opt='momentum', finetune_loss_func='softmax_cross_entropy',
                finetune_dropout=1)

            srbm.pretrain(trX, vlX)

            print('Start deep belief net finetuning...')
            srbm.fit(trX, trY, vlX, vlY)

            print('Test set accuracy: {}'.format(srbm.score(teX, teY)))

            def save_layers_output(which_set):
                if which_set == 'train':
                    trout = srbm.get_layers_output(trX)
                    for i, o in enumerate(trout):
                        np.save('/save/train' + '-layer-' + str(i + 1) + '-train', o)

                elif which_set == 'test':
                    teout = srbm.get_layers_output(teX)
                    for i, o in enumerate(teout):
                        np.save('/save/test' + '-layer-' + str(i + 1) + '-test', o)
                        # Save output from each layer of the model
            if True:
                print('Saving the output of each layer for the test set')
                save_layers_output('test')

            # Save output from each layer of the model
            if True:
                print('Saving the output of each layer for the train set')
                save_layers_output('train')

    def do_dbn_with_weight_matrix(self, path):
        dbn_model.compute_low_dimensions_data_matrix(
            utils.load_weight_matrix(path), self._input_data)


def trans_label_to_yadlt(datas):
    # 0:normal, 1:dos, 2:u2r, 3:probe
    _dict = {str(0.0): 0, str(float(1/3)): 1, str(float(2/3)): 2, str(1.0): 3}
    output_label_list = list()
    for j in range(len(datas)):
        data = datas[j]
        label = [0., 0., 0., 0.]
        label[_dict[str(data)]] = 1.
        output_label_list.append(label)
    return np.array(output_label_list)







