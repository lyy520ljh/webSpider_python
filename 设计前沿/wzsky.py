# -------------------------------------------------------------------------
#   程序：wzsky.py
#   版本：0.1
#   作者：lyy
#   日期：编写日期2019/01/05，测试日期2019/03/15
#   语言：Python 3.6.x
#   操作：python wzsky.py
#   功能：	爬取设计前沿-网页设计的图片存入mongodb
#
# -------------------------------------------------------------------------

from multiprocessing import Manager
from requests.exceptions import RequestException
from lxml import etree
import multiprocessing
import traceback
import requests
import pymongo
import time
import re

BASE_URL = 'http://www.wzsky.net'
FIRST_URL = None
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
    Chrome/72.0.3626.119 Safari/537.36',
    'Referer': 'http://www.wzsky.net'
}
# 各页面相关xpath语句
XPATH_INDEX = '//div[@class ="listimgtitle"]/a/@href'
XPATH_DETAIL_IMAGE = '//div[@class="info"]//img/@src'
XPATH_DETAIL_TITLE = '//h1[@class="title"]//text()'
XPATH_DETAIL_SOURCE = '//div[@class="source"]//text()'
# 发现下一页有多种语法，改为用正则
# XPATH_DETAIL_NEXT_URL = '//td[contains(@bgcolor,"")]//a/@href'


def get_page(url, flag=False):
    '''
    发起requests请求，获取网页
    :param url: 需要请求的url
    :param flag: 有些网页编码出现问题，增加flag区别
    :return: response.content or response.text
    '''
    try:
        response = requests.get(url, headers)
        # 网站编码为gb2312，但某些出现问题，改为用gb18030
        response.encoding = 'gb18030'
        content = response.content
        # flag判断返回response.text还是reponse.content
        # 原因是编码问题，例如索引页每页40条详细页url(个别除外),但获取的不足40条
        # 详细页也有类似的情况
        if flag:
            return response.text
        return content
    except RequestException as e:
        print(e)


def parse_index_page(url, flag=False):
    '''
    分析索引页，获取并返回详细页url
    :param url: 索引页url
    :param flag: 标记是否编码问题，是则获取response.text，否则response.content
    :return: xpath获取的结果
    '''
    try:
        text = get_page(url, flag)
        html = etree.HTML(text, etree.HTMLParser())
        result = html.xpath(XPATH_INDEX)
        return result
    except BaseException as e:
        print('index_url:', url, repr(e))


def parse_detail_page(detail_url, error_url_list, flag=False):
    '''
    分析详细页面，获取图片相关信息，返回结果
    :param detail_url: 从索引页获取的详细页面url
    :param error_url_list: 捕获到的错误url存储在error_url_list中
    :param flag: 标记是否为第一页
    :return:
    '''

    # BASE_URL: 'http://www.wzsky.net'
    url = BASE_URL + detail_url
    try:
        text = get_page(url)
        html = etree.HTML(text, etree.HTMLParser())
        image_url = html.xpath(XPATH_DETAIL_IMAGE)
        # next_url: False or url(eg:140744_2.html)
        next_url = find_with_re(text, url, 'td.*<a href=["\'](.*?)["\']>下一页</a>')
        if flag:
            title = html.xpath(XPATH_DETAIL_TITLE)
            sources = html.xpath(XPATH_DETAIL_SOURCE)
            # 繁体字获取不到，获取response.text解决（特殊情况，大约十条数据）
            if not sources:
                text = get_page(url, True)
                html = etree.HTML(text, etree.HTMLParser())
                sources = html.xpath(XPATH_DETAIL_SOURCE)
                title = html.xpath(XPATH_DETAIL_TITLE)
            sources_list = sources[0].replace('\u3000', ' ').split(' ', 2)
            origin, author, datetime = sources_list[0], sources_list[1], sources_list[2]
            # 如果有下一页，递归调用，注意此时flag为False
            if next_url:
                image_url.extend(parse_detail_page('/59/' + next_url, error_url_list))
            result = {
                '_id': url,
                'page_url': url,
                'image_url': image_url,
                'title': title[0],
                'origin': origin.split('：')[1],
                'author': author.split('：')[-1],
                'datetime': datetime.split('：')[1]
            }
            return result
        else:
            # 有下一页时，flag已经为False，进入这个else语句
            # 每个下一页只有图片，不需要重复获取标题等信息，只需返回image_url列表
            if next_url:
                image_url.extend(parse_detail_page('/59/' + next_url, error_url_list))
            return image_url
    except BaseException:
        # 抛出异常保存至error_url_list
        error_url_list.append({'_id': detail_url[:-5], 'URL': url, 'info': traceback.format_exc()})
        print(error_url_list[-1])


def find_with_re(content, url, pattern=''):
    '''
    判断是否存在下一页
    :param content: response.content
    :param pattern: 正则
    :return: 正则匹配的结果 or False
    '''
    try:
        find = re.search(pattern, str(content.decode('gb18030')))
    except:
        find = re.search(pattern, get_page(url, True))
    if find:
        return find.group(1)
    else:
        return False


# 定义数据库
client = pymongo.MongoClient('localhost', 27017)
spiderdb = client['spider']
# 保存url相关信息的collection
wzsky_image = spiderdb['wzsky_image']
# 保存错误url相关信息的collection
error_url = spiderdb['error_url']
# 已经保存到mongodb中的url信息列表
CHECK_LIST = list(wzsky_image.find())


def save_to_mongo(collection, result):
    '''
    保存至mongodb
    :param collection: 要保存的集合名称
    :param result: 要保存的结果
    :return:
    '''
    try:
        if collection.update({'_id': result['_id']}, {'$setOnInsert': result}, upsert=True):
            print('成功存储到:', collection.name)
    except Exception:
        print('存储失败')


def main(offset, error_url_list):
    '''
    :param offset: 索引页偏移量
    :param error_url_list: 错误url列表（包含错误信息），并保存到mongodb
    :return:
    '''
    index_url = 'http://www.wzsky.net/59/list_59_{}.html'.format(str(offset))
    detail_url_gen = parse_index_page(index_url)
    # 发现有些本该有40条的索引页获取的url不足40条
    if len(detail_url_gen) != 40:
        detail_url_gen_tmp = parse_index_page(index_url, True)
        detail_url_gen = detail_url_gen_tmp if len(detail_url_gen_tmp) > len(detail_url_gen) else detail_url_gen
    for detail_url in detail_url_gen:
        # 判断是否已经保存到mongodb中，若存在则不访问
        flag_mongo = False
        for d in CHECK_LIST:
            if BASE_URL+detail_url in d.values():
                flag_mongo = True
        if not flag_mongo:
            result = parse_detail_page(detail_url, error_url_list, flag=True)
            # print(result)
            if result:
                print('成功获取:', result['_id'])
                save_to_mongo(wzsky_image, result)
            time.sleep(1)
    # return index_url


if __name__ == '__main__':
    while True:
        inputs = input('请输入对应选项选择想要爬取的板块：\n\
58:平面设计\n\
59:网页设计\n\
60:广告设计\n\
61:三维动画\n\
62:标志设计\n\
exit/q:退出')
        if inputs == 'exit' or inputs == 'q':
            print('退出！')
            break
        elif inputs == '59' or inputs == '60' or inputs == '61' or inputs == '62':
            FIRST_URL = 'http://www.wzsky.net/%s/' % inputs
            break
        else:
            print('输入错误，请重新输入！')
    if FIRST_URL:
        # ERROR_URL_LIST作为进程共享变量保存抛出异常的url
        manager = Manager()
        ERROR_URL_LIST = manager.list()
        index_text = get_page(FIRST_URL)
        total_page = re.search(r'共.*?(\d+).*?页', str(index_text.decode('gb2312'))).group(1)
        # print(total_page)
        p_list = []
        # 开启多进程
        for i in range(1, int(total_page)+1):
            p = multiprocessing.Process(target=main, args=(str(i), ERROR_URL_LIST))
            p.start()
            p_list.append(p)
        for res in p_list:
            res.join()
        # print(ERROR_URL_LIST)
        # 将异常url保存至mongodb
        for err in ERROR_URL_LIST:
            save_to_mongo(error_url, err)
