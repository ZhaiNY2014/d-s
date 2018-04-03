#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import operator


class Preprocess:
    """
    input:
    [set_name,
      attrs[
          attr1[attr_name,attr_value[attr_value1, ...],
          attr2[attr_name,attr_value[attr_value1, ...],
          ...
      ],
      data[
           [,,,,],
           [,,,,],
           ...
          ]
    ]
    output:
    a list len = 122 and complete with [0,1]
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
    """
    def __init__(self, data):
        # print("init")
        self.data = data
        # dict 用于快速确定属性所在的位置
        self.dict = {}
        self.attr_min_list = init_output_list()
        self.attr_max_list = init_output_list()

    def do_preprocess(self):
        # print("dopp")
        set_name = self.data[0]
        attrs = self.data[1]
        datas = self.data[2]
        attr_list = []
        Preprocess.init_dict(self, attrs)
        pp_data_list = self.preprocess_data(datas)
        nor_data_list = self.normalize(pp_data_list)
        # print(len(output_list))
        with open("file/dataset_pp.txt", 'w') as f:
            for i in nor_data_list:
                f.write(i.__str__() + '\n')
        return nor_data_list

    def init_dict(self, attrs):
        # print("init dict")
        for i in range(len(attrs)):
            attr = attrs[i]
            if isinstance(attr[1], list):
                if operator.eq(attr[1], ['0', '1']):
                    self.dict[attr[0]] = i
                elif attr[0] == 'class':
                    self.dict[attr[0]] = i
                else:
                    for attr_s in attr[1]:
                        key = attr[0] + '.' + attr_s
                        self.dict[key] = i
            elif isinstance(attr[1], str):
                self.dict[attr[0]] = i
            else:
                print("【错误】：" + attr)

    def preprocess_data(self, datas):
        # print("doing ppdata")
        output_list = []
        attrs = self.data[1]
        for data in datas:  # data:一个41维的向量
            output = init_output_list()
            i = 0
            for value in data:  # value: 每一维的值
                attr = attrs[i]  # 取属性
                if isinstance(attr[1], list):
                    if attr[0] == 'class':
                        pass
                    else:
                        v = value
                        if operator.ne(attr[1], ['0', '1']):
                            v = 1
                        output[self.dict.get(attr[0] + '.' + value, -1)] = v
                else:
                    output[self.dict.get(attr[0], -1)] = value
                i = i + 1
            output_list.append(output)
        return output_list

    def normalize(self, datas):
        # print("doing normalize")
        outlists = []
        datas = self.normalize_str2num(datas)
        avg = normalize_get_avg_list(datas)
        for data in datas:
            outlist = init_output_list()
            for i in range(len(data)):
                value = data[i]
                if self.attr_max_list[i] != self.attr_min_list[i]:
                    outlist[i] = (value - avg[i]) / (self.attr_max_list[i] - self.attr_min_list[i])
                else:
                    outlist[i] = value - avg[i]
            outlists.append(outlist)
        return outlists

    def normalize_str2num(self, datas):
        # print("str to num")
        fdatas = []
        for data in datas:
            for i in range(len(data)):
                value = data[i]
                if isinstance(value, str):
                    data[i] = float(value)
                if self.attr_max_list[i] < data[i]:
                    self.attr_max_list[i] = data[i]
                if self.attr_min_list[i] > data[i]:
                    self.attr_min_list[i] = data[i]
            fdatas.append(data)
        return fdatas


def normalize_get_avg_list(datas):
    # print("get avg list")
    avg = init_output_list()
    sums = init_output_list()
    row = 0
    for data in datas:
        for i in range(len(data)):
            v = data[i]
            sums[i] = v + sums[i]
            print("get sum , now:" + i.__str__())
        row = row + 1
    if row != 0:
        for i in range(len(sums)):
            s = sums[i]
            avg[i] = s / row
            print("get avg, now :" + i.__str__())
    # print(avg)
    return avg


def init_output_list():
    output = []
    for i in range(123):
        output.append(0.0)
    return output
