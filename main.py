#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile
import preprocess
import dbn
import svm
import saveResult

if __name__ == "__main__":
    file_data = ReadFile.ReadFile("D:\\PycharmProjects\\DBN-SVM\\NSL_KDD-master").get_data()
    attr_name = file_data[0]
    train_data = file_data[1]
    test_data = file_data[2]
    attack_type = file_data[3]
    do_pp_train = preprocess.Preprocess(data=train_data)
    do_pp_test = preprocess.Preprocess(data=test_data)
    # print(data)
    do_dbn = dbn.DBN(data=do_pp_train)
    data = do_dbn.doDBN()
    do_pp_test = preprocess.Preprocess(data=test_data)
    do_svm = svm.SVM(train_data=do_pp_train, test_data=do_pp_test)
    data = do_svm.doSVM()
    do_save = saveResult.save(data)
