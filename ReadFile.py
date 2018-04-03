#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class ReadFile(object):
    def __init__(self, filePath):
        # 返回的list
        # [set_name,
        #   attrs[
        #       attr1[attr_name,attr_value[attr_value1, ...],
        #       attr2[attr_name,attr_value[attr_value1, ...],
        #       ...
        #   ],
        #   data[
        #        [,,,,],
        #        [,,,,],
        #        ...
        #       ]
        # ]
        set_name = ""
        attrs = []
        data = []
        self.list = [set_name, attrs, data]
        with open(filePath, 'r') as file:
            data_tag = False
            for line in file.readlines():
                if data_tag is False:
                    if line[0] == '@':
                        kind = line.split(' ', 1)
                        # 文件名
                        if kind[0].strip() == "@relation":
                            # set_name
                            self.list[0] = trim_mark(source=kind[1])
                        elif kind[0].strip() == "@attribute":
                            # attr
                            if '{' in kind[1] and '}' in kind[1]:
                                attr_name = trim_mark(kind[1].split(' ', 1)[0])
                                tmp_values = trim_brace(kind[1].split(' ', 1)[1]).split(',')
                                attr_values = []
                                for tmp_value in tmp_values:
                                    attr_value = tmp_value.split('\'')
                                    attr_values.append(attr_value[1])
                                attr = [attr_name, attr_values]
                                attrs.append(attr)
                            else:
                                attr = [trim_mark(kind[1].strip()), 'num']
                                attrs.append(attr)
                        elif kind[0].strip() == "@data":
                            data_tag = True
                elif data_tag is True:
                    data_values = line.strip().split(',')
                    data.append(data_values)

    def get_data(self):
        data = self.list
        return data


def trim_mark(source):
    source = source.strip()
    source = source.split('\'')[1]
    return source


def trim_brace(source):
    source.strip()
    source = source.strip('{')
    source = source.strip('}')
    return source
