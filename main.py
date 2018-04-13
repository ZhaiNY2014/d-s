#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile
import preprocess
import dbn
import svm
import saveResult

if __name__ == "__main__":
    read_train_data = ReadFile.ReadFile(filePath='D:/毕设相关/Original NSL KDD Zip/KDDTrain+_20Percent.arff')
    train_data = read_train_data.get_data()
    read_test_data = ReadFile.ReadFile(filePath='D:/毕设相关/Original NSL KDD Zip/KDDTest-21.arff')
    test_data = read_test_data.get_data()
    # print(data)
    do_pp_train = preprocess.Preprocess(data=train_data)
    data = do_pp_train.do_preprocess()
    # print(data)
    do_dbn = dbn.DBN(data=data)
    data = do_dbn.doDBN()
    do_pp_test = preprocess.Preprocess(data=test_data)
    do_svm = svm.SVM(train_data=train_data, test_data=test_data)
    data = do_svm.doSVM()
    do_save = saveResult.save(data)
