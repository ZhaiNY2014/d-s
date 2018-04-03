#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile
import preprocess
import dbn
import svm
import saveResult

if __name__ == "__main__":
    readfile = ReadFile.ReadFile(filePath='D:/毕设相关/Original NSL KDD Zip/KDDTrain+_20Percent.arff')
    data = readfile.get_data()
    # print(data)
    do_pp = preprocess.Preprocess(data=data)
    data = do_pp.do_preprocess()
    # print(data)
    do_dbn = dbn.DBN(data=data)
    data = do_dbn.doDBN()
    do_svm = svm.SVM(data=data)
    data = do_svm.doSVM()
    do_save = saveResult.save(data)
