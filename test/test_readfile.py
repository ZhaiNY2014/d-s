#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ReadFile

read = ReadFile.ReadFile("D:\\PycharmProjects\\DBN-SVM\\NSL_KDD-master").get_data()


def list2str(line):
    string = ""
    if isinstance(line, list):
        for a in line:
            if isinstance(a, list):
                string += ','.join(a)
            elif isinstance(a, str):
                string = string + ',' + a
    elif isinstance(line, str):
        string = line
    elif isinstance(line, dict):
        string = str(line)
    return string

with open("../file/dataset_rf.txt", 'w') as file:
    for i in range(len(read)):
        if i == 0:
            file.write('attr')
            for line in read[i]:
                file.write(line + '\n')
        if i == 1:
            file.write('train')
            for line in read[i]:
                file.write(list2str(line) + '\n')
        if i == 2:
            file.write('test')
            for line in read[i]:
                file.write(list2str(line) + '\n')
        if i == 3:
            file.write('dict')
            for line in read[i]:
                file.write(line.__str__())






