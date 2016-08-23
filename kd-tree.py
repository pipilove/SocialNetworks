#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'kd-tree'
__author__ = 'pika'
__mtime__ = '16-8-11'
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
import numpy as np


def split_select(_data):
    '''
    分裂结点选择程序：对于所有的样本点，统计它们在每个维上的方差，挑选出方差中的最大值，对应的维就是split域的值。返回分裂结点和分裂维度。
    '''
    data = np.array(_data)
    dim = data.shape[1]
    split = np.argmax([np.var(data[:, dim_i]) for dim_i in range(dim)])

    sort_data = sorted(data.tolist(), key=lambda x: x[split])
    dom_elt = sort_data[len(sort_data) // 2]
    return dom_elt, split


if __name__ == '__main__':
    tdata = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    dom_elt, split = split_select(tdata)
    print(dom_elt, split)

