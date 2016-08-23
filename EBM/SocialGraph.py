#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'pika'
__mtime__ = '16-8-17'
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
import linecache
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pandas import hashtable
import pickle
from pprint import pprint
from sklearn import preprocessing

from SocialNetworks.EBM import build_cooccu_vec
from SocialNetworks.EBM import build_userlocarray, calculate_diversity


def t2():
    gowalla_edges_filename = r'/media/pika/files/machine_learning/datasets/SocialNetworks/Gowalla/Gowalla_edges.txt'
    lines = linecache.getlines(gowalla_edges_filename)[:2000]
    gowalla_edges_filename = r'/media/pika/files/machine_learning/datasets/SocialNetworks/Gowalla/Gowalla_edges2000.txt'
    with open(gowalla_edges_filename, 'w') as gowalla_edges_file:
        gowalla_edges_file.write(''.join(lines))


def calculate_pathlen():
    '''
    计算所有user间的所有路径的路径长度
    :return: [[user0和user1的路径长度list], [u0, u2], ...]
    '''
    gowalla_edges_filename = r'/media/pika/files/machine_learning/datasets/SocialNetworks/Gowalla/Gowalla_edges2000.txt'
    edges = np.loadtxt(gowalla_edges_filename, dtype=int)
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (3, 6), (4, 6)]

    social_graph = nx.Graph()
    social_graph.add_edges_from(edges)
    # print(social_graph.edges())

    nodes = social_graph.nodes()

    # nx.draw_networkx(social_graph)
    # plt.show()

    # nx.all_simple_paths只能迭代一次
    paths_list = [nx.all_simple_paths(social_graph, nodes[i], nodes[j]) for i in range(len(nodes)) for j in
                  range(i + 1, len(nodes))]
    # pprint([list(i) for i in paths_list])
    paths_len_list = [[len(path) for path in paths] for paths in paths_list]
    # pprint(paths_len_list)
    return paths_len_list


def calculate_social_strength():
    '''
    通过social graph计算social strength
    '''

    # todo
    def jaccard_index():
        pass

    # todo
    def adamic_adar_similarity():
        pass

    def katz_score(epsilon):
        paths_len_list = calculate_pathlen()
        # pprint(paths_len_list)
        paths_len_series_list = [pd.value_counts(paths_len, sort=False) for paths_len in paths_len_list]
        pprint([dict(paths_len_series) for paths_len_series in paths_len_series_list])
        katz_score_list = [sum([epsilon ** k * v for k, v in paths_len_series.items()]) for paths_len_series
                           in paths_len_series_list]
        pprint(katz_score_list)
        return katz_score_list

    katz_score_list = katz_score(epsilon=10 ** -3)
    with open('katz_score_list', 'wb') as f:
        pickle.dump(katz_score_list, f)
        print("katz_score_list dump ended")


def calculate_parameter():
    '''
    social strength parameter calculation
    :return:
    '''
    with open('diversity_list', 'rb') as f:
        dij = np.array(pickle.load(f))
    with open('weighted_fre', 'rb') as f:
        fij = np.array(pickle.load(f))
    with open('katz_score_list', 'rb') as f:
        sij = np.array(pickle.load(f))
    # print(dij.shape, dij.dtype, type(dij))

    # fij = np.array([2.0, 4.0, 3.0])
    # dij = np.array([1.2, 3.3, 2.8])
    # sij = np.array([2.5, 4.0, 4.2])

    # 这里的dij, fij, sij应该算是features，所以应该是对features进行normalization
    # a = preprocessing.scale(fij.reshape(-1, 1))
    dij = preprocessing.scale(dij)
    fij = preprocessing.scale(fij)
    sij = preprocessing.scale(sij)

    sum_fij2 = sum(fij ** 2)
    sum_dij2 = sum(dij ** 2)
    sum_dij_fij = sum(dij * fij)
    sum_dij_sij = sum(dij * sij)
    sum_fij_sij = sum(fij * sij)
    denominator = sum_dij2 * sum_fij2 - sum_dij_fij ** 2

    alpha = (sum_fij2 * sum_dij_sij - sum_dij_fij * sum_fij_sij) / denominator
    beta = (sum_dij2 * sum_fij_sij - sum_dij_fij * sum_dij_sij) / denominator
    gamma = np.mean(sij) - alpha * np.mean(dij) - beta * np.mean(fij)
    print("alpha", alpha)
    print("beta", beta)
    print("gamma", gamma)
    return alpha, beta, gamma


if __name__ == '__main__':
    # t2()
    # exit()
    calculate_social_strength()
    # calculate_parameter()
