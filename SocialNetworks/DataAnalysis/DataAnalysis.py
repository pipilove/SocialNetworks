#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '时空数据分析'
__author__ = 'pika'
__mtime__ = '16-8-29'
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
import os
import sys
from pprint import pprint

from pyspark import SparkContext

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../.."))
try:
    from ..GlobalOptions import checkin_filename
    from ..Utils import read_data
except:
    from SocialNetworks.GlobalOptions import checkin_filename
    from SocialNetworks.Utils import read_data


def data_analysis():
    data = read_data(checkin_filename)
    data['check-in_time'] = data['check-in_time'].map(lambda x: x.hour)
    print(data.head(15))
    print(len(data['location_id']))
    loc_time_entropy = data.groupby(['location_id', 'check-in_time']).count()
    print(loc_time_entropy.head())
    # loc_time_entropy.to_csv(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'loc_time_entropy'))
    exit()
    for loc in loc_time_entropy:
        print(loc)
        exit()


def loc_time_entropy_fun():
    data_analysis()
    # loc_time_entropy = pd.read_csv(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'loc_time_entropy'),
    #                                header=0, skip_blank_lines=True)


def da_main():
    sc = SparkContext("local[4]", "Spark Data Analysis")
    # 将CSV格式的原始数据转化为(user,product,price)格式的记录集

    data = sc.textFile(checkin_filename).take(10)
    pprint(data)


if __name__ == '__main__':
    # da_main()
    loc_time_entropy_fun()

