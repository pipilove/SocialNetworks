#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'pika'
__mtime__ = '16-8-31'
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
                ┃  神兽保佑    ┣  ┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import pandas as pd


def read_data(filename, col_names=None):
    if col_names is None:
        col_names = ['user', 'check-in_time', 'latitude', 'longitude', 'location_id']
    lines = pd.read_csv(filename, sep='\t', header=None, names=col_names, parse_dates=[1],
                        skip_blank_lines=True)  # index_col=0
    lines = lines[['user', 'check-in_time', 'location_id']]
    # print("lines.head()\n", lines.head())
    return lines
