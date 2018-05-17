#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile
import preprocess
import dbn_model
from svm import model
import utils
from optparse import OptionParser


def main():
    root = utils.get_root_path(False)
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option('--learning_rate_rbm', action='store', type='string', dest='learning_rate_rbm')
    parser.add_option('--epochs_rbm', action='store', type='string', dest='epochs_rbm')
    parser.add_option('--batch_size', action='store', type='string', dest='batch_size')

    (opts, args) = parser.parse_args()

    file_data = ReadFile.ReadFile(root + '/NSL_KDD-master').get_data()
    data_pp = preprocess.Preprocess(file_data).do_predict_preprocess()
    dbn_model.DBN(data_pp).do_dbn('yadlt', opts=opts)
    dbn_model.DBN(data_pp).do_dbn_with_weight_matrix(root + '/save')
    model.do_svm()


if __name__ == '__main__':
    main()

