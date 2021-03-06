#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import dbn.tensorflow.models as dbn_model
import dbn.utils as utils

from yadlt.models.boltzmann import dbn
from yadlt.utils import utilities


class DBN(object):
    def __init__(self, dataset):
        self._input_data = dataset

    def do_dbn(self, action='pp', opts=None):
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

            trX, trY = np.array(self._input_data[0][0]), trans_label_to_yadlt(self._input_data[0][1])
            vlX, vlY = np.array(self._input_data[1][0][:10000]), trans_label_to_yadlt(self._input_data[1][1][:10000])
            teX, teY = np.array(self._input_data[1][0]), trans_label_to_yadlt(self._input_data[1][1])

            finetune_act_func = utilities.str2actfunc('relu')

            srbm = dbn.DeepBeliefNetwork(
                name='dbn', do_pretrain=True,
                rbm_layers=[100, 80, 50, 25, 5],
                finetune_act_func=finetune_act_func,
                rbm_learning_rate=[float(opts.learning_rate_rbm)] if opts is not None and opts.learning_rate_rbm is not None else [0.1],
                rbm_num_epochs=[int(opts.epochs_rbm)] if opts is not None and opts.epochs_rbm is not None else [100],
                rbm_gibbs_k=[1],
                rbm_gauss_visible=False, rbm_stddev=0.1,
                momentum=0.9,
                rbm_batch_size=[int(opts.batch_size)] if opts is not None and opts.batch_size is not None else [64],
                finetune_learning_rate=0.001,
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
                        np.save('train' + '-layer-' + str(i + 1) + '-train', o)

                elif which_set == 'test':
                    teout = srbm.get_layers_output(teX)
                    for i, o in enumerate(teout):
                        np.save('test' + '-layer-' + str(i + 1) + '-test', o)
                        # Save output from each layer of the model
            if False:
                print('Saving the output of each layer for the test set')
                save_layers_output('test')

            # Save output from each layer of the model
            if False:
                print('Saving the output of each layer for the train set')
                save_layers_output('train')

    def do_dbn_with_weight_matrix(self, path):
        dbn_model.compute_low_dimensions_data_matrix(
            utils.load_weight_matrix(path), self._input_data)


def trans_label_to_yadlt(datas):
    # 1:normal, 2:dos, 3:u2r, 4:probe 5:unknow
    _dict = {str(5): 5, str(1): 1, str(2): 2, str(3): 3, str(4): 4}
    output_label_list = list()
    for j in range(len(datas)):
        data = datas[j]
        label = [0., 0., 0., 0., 0.]
        label[_dict[str(data)] - 1] = 1.
        output_label_list.append(label)
    return np.array(output_label_list)







