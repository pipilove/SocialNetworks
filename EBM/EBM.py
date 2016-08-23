#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'EBM - An Entropy-Based Model to Infer Social Strength from Spatiotemporal Data'
__author__ = 'pika'
__mtime__ = '16-8-12'
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
import linecache
import math
import numpy as np
import pickle

from scipy import stats
import pandas as pd
import configparser
from statsmodels.sandbox.infotheo import renyientropy

sys.path.append()
from pyspark import SparkContext

if os.getlogin() == 'piting':
    SECTION = 'master'
elif os.getlogin() == 'pika':
    SECTION = 'dev'

conf = configparser.ConfigParser()
conf.read(r'./GlobalOptions.ini')
tmp_datadir = conf.get(SECTION, 'tmp_datadir')
os.makedirs(tmp_datadir, exist_ok=True)
print(os.getlogin())
exit()


def t():
    '''
    只读写部分数据
    '''
    checkin_filename = r'/media/pika/files/machine_learning/datasets/SocialNetworks/Gowalla/Gowalla_totalCheckins.txt'
    lines = linecache.getlines(checkin_filename)[:50000]
    checkin_outfilename = r'/media/pika/files/machine_learning/datasets/SocialNetworks/Gowalla/Gowalla_totalCheckins50000.txt'
    with open(checkin_outfilename, 'w') as checkin_file:
        checkin_file.write(''.join(lines))


def dist_test(lines):
    '''
    测试坐标点的距离，距离小的是不是同一个点
    '''
    from geopy import distance
    from scipy import spatial
    np.set_printoptions(precision=3, suppress=True, threshold=np.NaN)

    locs = lines[:, 2:4]
    dist_fun = lambda x, y: distance.vincenty(x, y).meters

    dist = spatial.distance.pdist(locs, metric=dist_fun)
    dist.sort()
    print(dist)


def time_test(lines):
    locs = lines
    for i in range(len(locs)):
        for j in range(i + 1, len(locs)):
            # if (0 < dist_fun(i[2:], j[2:]) < 10):
            if (locs[i][1] == locs[j][1]):
                print(i, j, locs[i], locs[j])


def build_userlocarray(lines):
    '''
    构建user-location的多维数组，元素如vi = (<t2>, <t3, t6> , ...)
    '''
    users = np.sort(lines['user'].unique())
    user_map = dict(zip(users, range(len(users))))
    locs = np.sort(lines['location_id'].unique())
    locs_map = dict(zip(locs, range(len(locs))))

    print('len(users), len(locs)', len(users), len(locs))
    userloc_array = np.zeros([len(users), len(locs)], dtype=np.ndarray)
    ut_index_df = lines.set_index(['user', 'location_id'])
    for ut_index in ut_index_df.index:
        userloc_array[user_map[ut_index[0]], locs_map[ut_index[1]]] = ut_index_df.loc[ut_index].values.reshape(-1)
        # if (ut_index == (0, 420315)):
        #     print(ut_index_df.loc[ut_index].values.reshape(-1))
    # print(userloc_array)
    return userloc_array


def build_cooccu_vec(userloc_array):
    '''
    构建co-occurrence vector
    '''
    co_occu_func = lambda x, y: [0 if i == 0 or j == 0 else len(set(i).intersection(set(j))) for i, j in zip(x, y)]
    co_occurs_list = [co_occu_func(userloc_array[i], userloc_array[j]) for i in range(len(userloc_array)) for j in
                      range(i + 1, len(userloc_array))]
    print(np.count_nonzero(np.array(co_occurs_list)))
    # print(np.array(co_occurs_list))
    return co_occurs_list


def calculate_diversity(co_occurs_list, q=0.1):
    '''
    compute Renyi Entropy-based Diversity
    '''
    co_occurs_array = np.array(co_occurs_list)

    def shannon_entropy(co_occurs_array):
        # shannon entropy
        shannon_entropy_list = [stats.entropy(ij / sum(ij), base=math.e) for ij in co_occurs_array]
        print("shannon_entropy_list:\n", shannon_entropy_list)
        shannon_diversity_list = [math.exp(i) for i in shannon_entropy_list]
        print("shannon_diversity_list:\n", shannon_diversity_list)
        print(shannon_diversity_list[0] / shannon_diversity_list[1])

    # shannon_entropy(co_occurs_array)

    def renyi_entropy(co_occurs_array):
        # Renyi Entropy-based Diversity
        diver_fun = lambda pij: ((pij ** q).sum()) ** (1 / (1 - q))
        renyi_diversity_list = [diver_fun(ij[np.nonzero(ij)] / sum(ij)) for ij in co_occurs_array]
        # print("renyi_diversity_list:\n", renyi_diversity_list)
        # print(renyi_diversity_list[0] / renyi_diversity_list[1])
        return renyi_diversity_list

    renyi_diversity_list = renyi_entropy(co_occurs_array)

    def renyi_entropy1(co_occurs_array):
        # Renyi Entropy-based Diversity
        renyi_entropy_list = [renyientropy(ij[np.nonzero(ij)] / sum(ij), alpha=q, logbase=math.e) for ij in
                              co_occurs_array]
        print("renyi_entropy_list:\n", renyi_entropy_list)
        renyi_diversity_list = [math.exp(i) for i in renyi_entropy_list]
        print("renyi_diversity_list:\n", renyi_diversity_list)
        print(renyi_diversity_list[0] / renyi_diversity_list[1])
        # return renyi_diversity_list

    # renyi_diversity_list = renyi_entropy1(co_occurs_array)
    return renyi_diversity_list


def calculate_locentropy(userloc_array):
    '''
    计算位置熵
    '''
    userloc_cnt_array = np.vectorize(lambda x: 0 if x == 0 else len(x))(userloc_array)
    userloc_cnt_array = np.zeros([22, 5], dtype=int)
    # print(userloc_cnt_array)
    userloc_cnt_array[:, [0, 4]] = 10
    userloc_cnt_array[0:2] = [[10, 1, 0, 0, 9], [2, 3, 2, 2, 3]]
    # print(userloc_cnt_array)

    pul = userloc_cnt_array / userloc_cnt_array.sum(axis=0)

    f = np.vectorize(lambda x: 0 if x == 0 else x * math.log(x))
    loc_entropy = (lambda x: -np.sum(f(x), axis=0))(pul)
    # print("loc_entropy:\n", loc_entropy)
    return loc_entropy


def calculate_weightfre(loc_entropy, co_occurs):
    '''
    compute weighted frequency
    '''
    weight_fre_fun = lambda x: sum([c * math.exp(-l) for c, l in zip(x, loc_entropy)])
    weight_fre_list = [weight_fre_fun(c) for c in co_occurs]
    # print("weight_fre_list:\n", weight_fre_list)
    return weight_fre_list


def predict_social_strength(div, fre, alpha=0.5, beta=0.5, gamma=0):
    '''
    计算social strength
    '''
    return alpha * div + beta * fre + gamma


if __name__ == '__main__':
    # t()
    # exit()

    checkin_filename = conf.get(SECTION, 'checkin_filename')
    userloc_array_filename = conf.get(SECTION, 'userloc_array_filename')
    co_occurs_list_filename = conf.get(SECTION, 'co_occurs_list_filename')
    diversity_list_filename = conf.get(SECTION, 'diversity_list_filename')
    weighted_fre_filename = conf.get(SECTION, 'weighted_fre_filename')

    col_names = ['user', 'check-in_time', 'latitude', 'longitude', 'location_id']
    lines = pd.read_csv(checkin_filename, sep='\t', header=None, names=col_names, parse_dates=[1],
                        skip_blank_lines=True)  # index_col=0
    lines = lines[['user', 'check-in_time', 'location_id']]
    print("lines.head()\n", lines.head())
    # dist_test(lines.values)
    # time_test(lines.values)

    userloc_array = build_userlocarray(lines)
    with open(userloc_array_filename, 'wb') as f:
        pickle.dump(userloc_array, f)
        print("userloc_array dump ended")
    with open('userloc_array', 'rb') as f:
        userloc_array = pickle.load(f)

    co_occurs_list = build_cooccu_vec(userloc_array)
    with open('co_occurs_list', 'wb') as f:
        pickle.dump(co_occurs_list, f)
        print("co_occurs_list dump ended")
    with open('co_occurs_list', 'rb') as f:
        co_occurs_list = pickle.load(f)
    # co_occurs_list = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 2, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, 0, 0]]
    co_occurs_list = [[10, 1, 0, 0, 9], [2, 3, 2, 2, 3]]

    diversity_list = calculate_diversity(co_occurs_list, q=0.1)
    with open('diversity_list', 'wb') as f:
        pickle.dump(diversity_list, f)
        print("diversity_list dump ended")
    with open('diversity_list', 'rb') as f:
        diversity_list = pickle.load(f)

    loc_entropy = calculate_locentropy(userloc_array)
    weighted_fre = calculate_weightfre(loc_entropy, co_occurs_list)
    with open('weighted_fre', 'wb') as f:
        pickle.dump(weighted_fre, f)
        print("weighted_fre dump ended")
    with open('weighted_fre', 'rb') as f:
        weighted_fre = pickle.load(f)

    predict_social_strength(diversity_list, weighted_fre, alpha=0.483, beta=0.520, gamma=0)
    # xyt = lines[:, 1:4]
    # print(xyt)
    # KDTree(xyt)
