#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile
import preprocess
import dbn_model
from svm import model
import os

if __name__ == "__main__":
    path_cur = os.path.abspath('.')
    path_pre = os.path.abspath('..')
    file_data = ReadFile.ReadFile(path_cur + '/NSL_KDD-master').get_data()
    data_pp = preprocess.Preprocess(file_data).do_predict_preprocess()
    dbn_model.DBN(data_pp).do_dbn('yadlt')
    dbn_model.DBN(data_pp).do_dbn_with_weight_matrix(path_cur + '/save')
    do_svm = model.do_svm()
