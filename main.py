#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile
import preprocess
import dbn_model
from svm import svm

if __name__ == "__main__":
    file_data = ReadFile.ReadFile("D:\\PycharmProjects\\DBN-SVM\\NSL_KDD-master").get_data()
    data_pp = preprocess.Preprocess(file_data).do_preprocess()
    do_dbn = dbn_model.DBN(data_pp).do_dbn()
    data = do_dbn.doDBN()
    do_pp_test = preprocess.Preprocess(data=test_data)
    do_svm = svm.SVM(train_data=do_pp_train, test_data=do_pp_test)
    data = do_svm.doSVM()
    do_save = saveResult.save(data)
