#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import operator
import numpy as np


class Preprocess:
    """
    input:
    list[
        attr_name[],
        train_data[],
        test_data[],
        attack_type[]
        ]
    output:
    a tuple include (train_data, test_data), and both are tuple like (data, targit),data is a list len = 122 
    and complete with 0-1 normalize:
    [duration, protocol_type.tcp, protocol_type.udp, protocol_type.cmp, service.aol, service.auth, service.bgp, 
    service.courier, service.csnet_ns, service.ctf, service.daytime, service.discard, service.domain, service.domain_u, 
    service.echo, service.eco_i, service.ecr_i, service.efs, service.exec, service.finger, service.ftp, 
    service.ftp_data, service.gopher, service.harvest, service.hostnames, service.http, service.http_2784, 
    service.http_443, service.http_8001, service.imap4, service.IRC, service.iso_tsap, service.klogin, service.kshell, 
    service.ldap, service.link, service.login, service.mtp, service.name, service.netbios_dgm, service.netbios_ns, 
    service.netbios_ssn, service.netstat, service.nnsp, service.nntp, service.ntp_u, service.other, service.pm_dump, 
    service.pop_2, service.pop_3, service.printer, service.private, service.red_i, service.remote_job, service.rje, 
    service.shell, service.smtp, service.sql_net, service.ssh, service.sunrpc, service.supdup, service.systat, 
    service.telnet, service.tftp_u, service.tim_i, service.time, service.urh_i, service.urp_i, service.uucp, 
    service.uucp_path, service.vmnet, service.whois, service.X11, service.Z39_50, flag.OTH, flag.REJ, 
    flag.RSTO, flag.RSTOS0, flag.RSTR, flag.S0, flag.S1, flag.S2, flag.S3, flag.SF, flag.SH, src_bytes, dst_bytes, 
    land, wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, 
    num_root ,num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count,
     srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, 
     srv_diff_host_rate, dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, 
     dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, 
     dst_host_rerror_rate, dst_host_srv_rerror_rate]
     targit is a list mapping 4 types attack_type 
    """
    def __init__(self, data):
        self._train_data = data[1]
        self._test_data = data[2]
        self._attr_name = data[0]
        self._attack_type = data[3]

    def do_preprocess(self):
        self._init_dict()
        # a tuple(train_data,train_targit), label #0 is data and #1 is targit
        train_pp = self._preprocess_data(self._train_data)
        # a tuple(test_data,test_targit)
        test_pp = self._preprocess_data(self._test_data)
        train_n = _normalize(train_pp)
        test_n = _normalize(test_pp)

        return_dataset = (train_n, test_n)
        return return_dataset

    def _init_dict(self):
        self.dict = {}
        prefix = ""
        index = 0
        for i in range(len(self._attr_name)):
            attr = self._attr_name[i]
            if attr.__contains__('.'):
                if attr.split('.')[0] == prefix:
                    pass
                else:
                    prefix = attr.split('.')[0]
                    self.dict[index] = i
                    index += 1
            else:
                prefix = attr
                self.dict[index] = i
                index += 1

    def _preprocess_data(self, X):
        _input_data = X
        datas = list()
        datas_expend = list()
        targit = list()
        # 分离data和targit
        for data in _input_data:
            _datas = data[:41]
            _targit = self._attack_type[data[42]]
            datas.append(_datas)
            targit.append(_targit)
        # 处理data
        for data in datas:
            data_expend = _init_output_list()
            for i in range(len(data)):
                if i == 1 or i == 2 or i == 3:  # 字符型需映射
                    for j in range(self.dict[i], self.dict[i+1]):
                        if data[i] == self._attr_name[j].split('.')[1]:
                            data_expend[j] = 1.0
                else:  # 数值型，无需映射
                    data_expend[self.dict[i]] = float(data[i])
            datas_expend.append(data_expend)
        # 处理targit
        _dict_targit = {'normal': 0, 'dos': 1, 'u2r': 2, 'r2l': 3, 'probe': 4, 'unknown': 5}
        targit_class = list()
        for i in range(len(targit)):
            targit_class.append(_dict_targit[targit[i]])

        return datas_expend, targit_class


def _normalize(X):
    _datas = X[0]
    output = list()
    data_a = np.array(_datas)
    _max = np.amax(data_a, 0).tolist()
    _min = np.amin(data_a, 0).tolist()
    for data in _datas:
        outrow = _init_output_list()
        for i in range(len(data)):
            if _max[i] - _min[i] != 0.:
                value = (data[i] - _min[i]) / (_max[i] - _min[i])
            else:
                value = 0. if _max[i] == 0.0 else 1.
            outrow[i] = value
        output.append(outrow)

    return output, X[1]

    # def _normalize_str2num(self, datas):
    #     # print("str to num")
    #     fdatas = []
    #     for data in datas:
    #         for i in range(len(data)):
    #             value = data[i]
    #             if isinstance(value, str):
    #                 data[i] = float(value)
    #             if self.attr_max_list[i] < data[i]:
    #                 self.attr_max_list[i] = data[i]
    #             if self.attr_min_list[i] > data[i]:
    #                 self.attr_min_list[i] = data[i]
    #         fdatas.append(data)
    #     return fdatas


def _init_output_list():
    output = []
    for i in range(122):
        output.append(0.0)
    return output
