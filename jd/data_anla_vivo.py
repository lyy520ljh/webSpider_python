from reportPPT import *
from snownlp import SnowNLP

import pandas as pd
from review import reviews
import numpy as np

f = open('clean_csv/vivoX21i.csv',encoding='utf-8')
data = pd.read_csv(f)
print('包含字段：\n'+'||'.join(data.columns))

data['creationTime'] = pd.to_datetime(data['creationTime'])
des1 = data[['days','score']].describe()
des2 = data[['content','productColor']].describe()
# print(des1)
# print(des2)
cols = [content for content in data['content'] if content != '此用户未填写评价内容' and content != '此用户未及时填写评价内容，系统默认好评！']
data=data[data.content.isin(cols)]
comments = reviews(texts=data['content'], scores=data['score'], creationTime=data['creationTime'])
print(comments.describe())
comments_describe1 = comments.describe()

# 创建pptx文档类
from pptx import Presentation
prs = Presentation()
# 添加第一页
add_para(prs, 'summary before analysis', paragraph='\n'.join('{}:{}'.format(k, comments_describe1[k]) for k in comments_describe1.index))
# print(comment)

comments.seg(product_dict='dict/mobile_dict.txt', stopwords='dict/chinese.txt', add_words=['极夜黑', '宝石红', '极光白'])
initial_words = ['............................................................................']
comments.del_invalid(initial_words=initial_words, max_rate=0.6)
print(comments.describe())
comments_describe2 = comments.describe()

# 添加第二页
add_para(prs, 'summary after analysis', paragraph='\n'.join('{}:{}'.format(k, comments_describe2[k]) for k in comments_describe2.index))

# 添加第3-8页
for k in ['好评','中评','差评']:
    keywords = comments.get_kws(comments.scores == k)
    # print('{}的关键词为：'.format(k)+'|'.join(keywords))
    topics = comments.find_topic(comments.scores==k,n_topic=5)
    add_para(prs, k, text='{}的关键词为：'.format(k) + ' | '.join(keywords), paragraph='\n'.join(['主题{}:{}'.format(i, ' | '.join([k[0] for k in topic[1]])) for i, topic in enumerate(topics)]))
    comments.find_topic(comments.scores==k,n_topic=5)
    filename = 'vivo wordcloud of {}'.format(k)
    filename = ' '.join(SnowNLP(filename).pinyin)
    comments.get_wordcloud(comments.scores == k, filename=filename)
    add_pic(prs, k + '词云', 'png/' + filename + '.png')
    print('='*20)



features = comments.get_features(min_support=0.005)
print(np.array(features).T)
texts_fw = comments.find_kws('外观')
print(texts_fw.head(15))

features_new = list(set(features)-set(['快递','物流','有点','下单','速度','京东']))
print(np.array(features_new).T)
features_new_array = np.array(features_new).T
add_para(prs, '挖掘出的特征', paragraph=' | '.join(f for f in features_new_array))

features_option,features_corpus = comments.features_sentiments(features_new,method='score')
print(features_option.sort_values('mention_count',ascending=False))

print('好评占比大于所有样本（好评率95.03%）的特征：')
print(features_option[(features_option['p_positive']>=0.9503)].index)
print('好评占比小于所有样本（好评率95.03%）的特征：')
print(features_option[(features_option['p_positive']<0.9503)].index)
indxs = list(features_option[(features_option['p_positive']<0.9503)].index)
add_para(prs, '好评占比小于所有样本（好评率95.03%）的特征', paragraph=' | '.join(indx for indx in indxs))

for indx in indxs:
    print('='*20+indx+'='*20)
    print('//'.join(comments.find_kws(indx).head(15)))
    add_para(prs, '=' * 3 + '\"' + indx + '\"' + '对应的语料' + '=' * 3, paragraph=' // '.join(comments.find_kws(indx).head(15)))

prs.save('report/京东vivoX21i评论分析报告.pptx')