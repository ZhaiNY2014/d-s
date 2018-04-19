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
                train_data.append(line.strip().split(','))

        with open(pakage_path + '/KDDTest+.csv') as test_file:
            for line in test_file.readlines():
                test_data.append(line.strip().split(','))

        attr_str = "duration, protocol_type.tcp, protocol_type.udp, protocol_type.cmp, service.aol, "+\
                   "service.auth, service.bgp, service.courier, service.csnet_ns, service.ctf, service.daytime, " +\
                   "service.discard, service.domain, service.domain_u, service.echo, service.eco_i, service.ecr_i, " +\
                   "service.efs, service.exec, service.finger, service.ftp, service.ftp_data, service.gopher, " +\
                   "service.harvest, service.hostnames, service.http, service.http_2784, service.http_443, " +\
                   "service.http_8001, service.imap4, service.IRC, service.iso_tsap, service.klogin, service.kshell, " +\
                   "service.ldap, service.link, service.login, service.mtp, service.name, service.netbios_dgm, " +\
                   "service.netbios_ns, service.netbios_ssn, service.netstat, service.nnsp, service.nntp, " +\
                   "service.ntp_u, service.other, service.pm_dump, service.pop_2, service.pop_3, service.printer, " +\
                   "service.private, service.red_i, service.remote_job, service.rje, service.shell, service.smtp, " +\
                   "service.sql_net, service.ssh, service.sunrpc, service.supdup, service.systat, service.telnet, " +\
                   "service.tftp_u, service.tim_i, service.time, service.urh_i, service.urp_i, service.uucp, " +\
                   "service.uucp_path, service.vmnet, service.whois, service.X11, service.Z39_50, flag.OTH, flag.REJ, " +\
                   "flag.RSTO, flag.RSTOS0, flag.RSTR, flag.S0, flag.S1, flag.S2, flag.S3, flag.SF, flag.SH, " +\
                   "src_bytes, dst_bytes, land, wrong_fragment, urgent, hot, num_failed_logins, logged_in, " +\
                   "num_compromised, root_shell, su_attempted, num_root ,num_file_creations, num_shells, " +\
                   "num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count, srv_count, serror_rate, " +\
                   "srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, " +\
                   "dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, " +\
                   "dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, " +\
                   "dst_host_srv_serror_rate,dst_host_rerror_rate, dst_host_srv_rerror_rate"
        for attr in attr_str.split(','):
            attr_name.append(attr.strip())

        with open(pakage_path + '/Attack Types.csv') as attack_type_file:
            lines = attack_type_file.readlines()
            for i, line in zip(list(range(len(lines))), lines):
                attack_type[str(i)] = line.split(',')[1].strip()

        self._return_data_set = [attr_name, train_data, test_data, attack_type]

    def get_data(self):
        data = self._return_data_set
        return data
