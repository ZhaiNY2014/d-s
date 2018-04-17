#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class ReadFile(object):
    def __init__(self, pakage_path):
        """
        return_list[
            attr_name[],
            train_data[],
            test_data[],
            attack_type[]
        ]
        """
        train_data = []
        test_data = []
        attr_name = []
        attack_type = {}

        with open(pakage_path + '/KDDTrain+.csv') as train_file:
            for line in train_file.readlines():
                train_data.append(line.strip())

        with open(pakage_path + '/KDDTest+.csv') as test_file:
            for line in test_file.readlines():
                test_data.append(line.strip())

        with open(pakage_path + '/Field Names.csv') as field_name:
            for line in field_name.readlines():
                _line = line.split(',')
                if _line[1].strip() == 'continuous':
                    attr_name.append(_line[0])
                elif _line[1].strip() == 'symbolic':
                    attr_name.append(list().append(_line[0]))

        with open(pakage_path + '/Attack Types.csv') as attack_type_file:
            lines = attack_type_file.readlines()
            for i, line in zip(list(range(len(lines))), lines):
                attack_type[str(i)] = line.split(',')[1].strip()

        self.return_data_set = [attr_name, train_data, test_data, attack_type]

    def get_data(self):
        data = self.return_data_set
        return data
