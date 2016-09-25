#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '读取配置文件进行全局变量设置'
__author__ = 'pika'
__mtime__ = '16-8-23'
__email__ = 'pipisorry@126.com'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import configparser
import os
import sys
import pwd

if pwd.getpwuid(os.geteuid()).pw_name == 'piting':
    SECTION = 'master'
elif pwd.getpwuid(os.geteuid()).pw_name == 'pika':
    SECTION = 'dev_pika'
elif pwd.getpwuid(os.geteuid()).pw_name == 'pipi':
    SECTION = 'dev_pipi'
else:
    print("please set like wd.getpwuid(os.geteuid()).pw_name == 'pipi'")
    sys.exit()

conf = configparser.ConfigParser()
conf.read(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'GlobalOptions.ini'))

# spark environment settings
os.environ['SPARK_HOME'] = conf.get(SECTION, 'SPARK_HOME')
sys.path.append(os.path.join(conf.get(SECTION, 'SPARK_HOME'), 'python'))
os.environ["PYSPARK_PYTHON"] = conf.get(SECTION, 'PYSPARK_PYTHON')
os.environ['SPARK_LOCAL_IP'] = conf.get(SECTION, 'SPARK_LOCAL_IP')

# io var
checkin_filename = conf.get(SECTION, 'checkin_filename')
gowalla_edges_filename = conf.get(SECTION, 'gowalla_edges_filename')
