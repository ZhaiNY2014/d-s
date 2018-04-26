#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile
import preprocess
import dbn_model
import saveResult
from svm import model

if __name__ == "__main__":
    file_data = ReadFile.ReadFile("D:\\PycharmProjects\\DBN-SVM\\NSL_KDD-master").get_data()
    data_pp = preprocess.Preprocess(file_data).do_preprocess()
    data_dbn = dbn_model.DBN(data_pp).do_dbn()
    do_svm = model.SVM(train_data=data_dbn[0], test_data=data_dbn[1])
    data = do_svm.doSVM()
    do_save = saveResult.save(data)
