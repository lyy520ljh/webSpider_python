import pandas as pd
f = open('clean_csv/华为荣耀 V10.csv',encoding='utf-8')
data = pd.read_csv(f)
print('包含字段：\n'+'||'.join(data.columns))

data['creationTime'] = pd.to_datetime(data['creationTime'])
des1 = data[['score']].describe()
des2 = data[['content','productColor']].describe()
print(des1)
print(des2)


cols = [content for content in data['content'] if content != '此用户未填写评价内容' and content != '此用户未及时填写评价内容，系统默认好评！']
data=data[data.content.isin(cols)]
from review import reviews
comments = reviews(texts=data['content'], scores=data['score'], creationTime=data['creationTime'])



comments.seg(product_dict='dict/mobile_dict.txt', stopwords='dict/chinese.txt', add_words=['幻夜黑', '魅丽红', '极光蓝', '沙滩金', '炫影蓝'])
initial_words = ['凑够','字数']
comments.del_invalid(initial_words=initial_words, max_rate=0.6)
print(comments.describe())
comments_describe2 = comments.describe()


for k in ['好评','中评','差评']:
    keywords = comments.get_kws(comments.scores == k)
    print('{}的关键词为：'.format(k)+'|'.join(keywords))
    # topics = comments.find_topic(comments.scores==k,n_topic=5)



import numpy as np
features = comments.get_features(min_support=0.005)
print(np.array(features).T)
texts_fw = comments.find_kws('摄像头')
print(texts_fw.head(15))

features_new = list(set(features)-set(['老公','玩王','物流','快递','时间','东西']))
print(np.array(features_new).T)
features_new_array = np.array(features_new).T


features_option,features_corpus = comments.features_sentiments(features_new,method='score')
print(features_option.sort_values('mention_count',ascending=False))


print('好评占比小于所有样本（好评率96.84%）的特征：')
print(features_option[(features_option['p_positive']<0.9684)].index)
indx1 = list(features_option[(features_option['p_positive']<0.9684)].index)

print('非好评占比大于所有样本（非好评率3.16%）的特征：')
print(features_option[(features_option['p_negative']>=0.0316)].index)
indx2 = list(features_option[(features_option['p_negative']>=0.0316)].index)

print([x for x in indx1 if x in indx2])

# # coding:utf-8
# from multiprocessing.dummy import Pool
# import requests
# import json
# import re
# import sys
# import os
# import csv
# from lxml.zlyhtml import fromstring
# from lxml import _elementpath
# import pandas as pd
#
# if not os.path.exists("files_iphone"):
#     os.makedirs('files_iphone')
#
# path = sys.path[0]
# print(path)
# if os.path.isdir(path):
#     mypath = path
# if os.path.isfile(path):
#     mypath = os.path.dirname(path)
# else:
#     mypath = os.getcwd()
# print(mypath)
# filepath = mypath + '\\files_iphone\\'  # 文件保存路径
#
# print(filepath)