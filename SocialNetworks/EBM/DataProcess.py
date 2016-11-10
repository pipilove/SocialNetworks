#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'checkin数据处理'
__author__ = 'pipi'
__mtime__ = '11/4/16'
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


def validateTest(line, flag=True):
    '''
    测试数据行是否合法可用。合法示例：
    userID	Time(GMT)	                    VenueId	            VenueName	VenueLocation	VenueCategory
    1	    Sat Jul 30 20:15:24 +0000 2011	4d8933fc6daeb60c31b04ae0	{Bosie Tea Parlor}	{40.731354990929475,-74.00363118575608,New York,NY,United States}	{Food,}
    '''
    import traceback
    import dateutil.parser
    try:
        user_id, time, venue_id, venue_name, venue_loc, venue_cat = line.split('\t')
    except:
        print("line split error\n{}".format(line))
        # print(traceback.format_exc())
        return False

    if type(user_id) is str and not user_id.isdigit():
        print("user_id: {} error".format(user_id))
        return False

    try:
        if type(dateutil.parser.parser(time)) is not dateutil.parser.parser:
            print("time: {} error".format(time))
            return False
    except:
        # print(traceback.format_exc())
        print("time: {} error".format(time))
        return False

    if len(venue_id) < 1 or len(venue_name.strip('{|}')) < 1:
        print("venue_id or venue_name error: \n {}".format(line))
        return False

    longitude, latitude, loc_des = venue_loc.strip('{|}').split(',', maxsplit=2)
    try:
        float(longitude)
        float(latitude)
    except:
        # print(traceback.format_exc())
        print("longitude or latitude error: \n {}".format(line))
        return False

    if len(venue_cat) < 1:
        print("venue_cat error: \n {}".format(line))
        return False

    return flag


def processLine(line):
    '''
    对数据行进行处理
    userID	Time(GMT)	                    VenueId	            VenueName	VenueLocation	VenueCategory
    1	    Sat Jul 30 20:15:24 +0000 2011	4d8933fc6daeb60c31b04ae0	{Bosie Tea Parlor}	{40.731354990929475,-74.00363118575608,New York,NY,United States}	{Food,}
    ===== >
    userID	Time(GMT)	                    VenueLocation	                    VenueId	VenueWords  VenueCategory
    1	    Sat Jul 30 20:15:24 +0000 2011	40.731354990929475,-74.00363118575608	4d8933fc6daeb60c31b04ae0   Bosie Tea Parlor,New York,NY,United States   Food
    '''
    user_id, time, venue_id, venue_name, venue_loc, venue_cat = line.strip().split('\t')
    longitude, latitude, loc_des = venue_loc.strip('{|}').split(',', maxsplit=2)
    venue_loc = ','.join([longitude, latitude])
    venue_words = ','.join([venue_name.strip('{|}'), loc_des])
    venue_cat = venue_cat.strip('{|}').strip(',')
    line = '\t'.join([user_id, time, venue_loc, venue_id, venue_words, venue_cat])
    # print(line)
    return line


def dataPreprocess():
    import sys, os, subprocess
    sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../..'))
    from SocialNetworks.SocialNetworks.GlobalOptions import CA_DATASET_DIR, checkin_ca

    # df = pd.read_csv(checkin_ca, header=0, skip_blank_lines=True, sep='\t', parse_dates=['Time(GMT)']).dropna()
    subprocess.run("dos2unix -k '" + os.path.join(CA_DATASET_DIR, checkin_ca) + "'", shell=True)
    subprocess.run(r"mv '" + os.path.join(CA_DATASET_DIR, checkin_ca) + "' /tmp", shell=True)
    with open(os.path.join('/tmp', checkin_ca), encoding='utf-8') as infile, open(
            os.path.join(CA_DATASET_DIR, checkin_ca), 'w', encoding='utf-8') as outfile:
        # outfile.write(infile.readline().strip() + '\n')
        infile.readline()
        outfile.write(
            '\t'.join(['userID', 'Time(GMT)', 'VenueLocation', 'VenueId', 'VenueWords', 'VenueCategory']) + '\n')
        for lid, line in enumerate(infile):
            if validateTest(line):
                line = processLine(line)
                outfile.write(line + '\n')


def buildLocKDTree():
    '''
    构建位置的kdtree
    '''
    import pandas as pd
    import numpy as np
    from sklearn import neighbors
    import sys, os, pickle
    sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../..'))
    from SocialNetworks.SocialNetworks.GlobalOptions import CA_DATASET_DIR, checkin_ca

    df = pd.read_csv(os.path.join(CA_DATASET_DIR, checkin_ca), header=0, sep='\t')
    # l_array = df['VenueLocation'].map(lambda s: np.array(s.split(','))).values.reshape([-1,2])
    l_list = df['VenueLocation'].values.tolist()
    l_array = np.array([s.split(',') for s in l_list]).astype(np.float64)

    loc_kdtree = neighbors.KDTree(l_array, metric='minkowski')

    pickle.dump(loc_kdtree, open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'loc_kdtree'), 'wb'))

    def queryTest():
        inds = loc_kdtree.query(l_array[0].reshape(1, -1), k=5, return_distance=False)
        print(inds)
        print(l_array[0])
        # np.set_printoptions(precision=15)
        print(l_array.astype(str)[inds])

    # queryTest()
    return loc_kdtree


if __name__ == '__main__':
    # dataPreprocess()
    buildLocKDTree()
