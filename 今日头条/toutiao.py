# -------------------------------------------------------------------------
#   程序：toutiao.py
#   版本：0.2
#   作者：lyy
#   日期：编写日期2019/01/28
#   语言：Python 3.6.x
#   操作：python toutiao.py
#   功能：	爬取今日头条图片频道
#
# -------------------------------------------------------------------------

import re
import time
import json
# import hashlib
# import execjs
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from requests.exceptions import RequestException
from selenium import webdriver
# browser = webdriver.Chrome()
option = webdriver.ChromeOptions()
option.add_argument("headless")
browser = webdriver.Chrome(chrome_options=option)


proxy = {'http':'http://59.62.164.141:9999'}
headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, sdch",
    "Accept-Language": "zh-CN,zh;q=0.8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36",
}

# requests访问
def get_page(url, header):
    try:
        response = requests.get(url, header, verify=False, proxies=proxy)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        print('索引页出错', url)
        return None


# 解析json数据，返回图片页url
def parse_json_page(html):
    json_data = json.loads(html)
    if json_data and 'data' in json_data.keys():
        for item in json_data.get('data'):
            item_id = item.get('item_id')
            if item_id:
                yield "https://www.toutiao.com/a" + item_id
            # # source_url = item.get('source_url')
            # if source_url:
            #     yield 'https://www.toutiao.com' + source_url


# source_url会重定向，使用selenium可以解决
def get_page_detail(url):
    try:
        browser.get(url)
        browser.execute_script('window.open()')
        return browser.page_source
    finally:
        browser.close()
        browser.switch_to.window(browser.window_handles[0])


# 解析图片页，获取图片数据
def parse_page_detail(html, url):
    soup = BeautifulSoup(html, 'lxml')
    # print(html)
    title = soup.select('title')[0].get_text()
    print(title)
    images_pattern = re.compile('gallery: JSON.parse\("(.*?),\\\\"max_img_width', re.S)
    result = re.search(images_pattern, html)
    if result:
        result = result.group(1).replace(r'\"', '"').replace('\/', '/') + '}'
        print(result)
        data = json.loads(result)
        if data and 'sub_images' in data.keys():
            sub_images = data.get('sub_images')
            images = [item.get('url') for item in sub_images]
            return {
                'title': title,
                'url': url,
                'image': images
            }


# 访问图片频道分组首页，获取json数据url
def get_json_page(keyword, max_behot_time):
    # params = json.loads(get_js())
    # print(params)
    format_str = keyword
    if keyword == '组图':
        format_str = 'news_image'
    first_url = 'https://www.toutiao.com/ch/{}/'.format(format_str)
    browser.get(first_url)
    params = browser.execute_script('return ascp.getHoney()')
    eas, ecp = params['as'], params['cp']
    _signature = browser.execute_script('return TAC.sign({})'.format(str(max_behot_time)))
    # print(eas, ecp)
    # print(_signature)

    # cookie = browser.get_cookies()
    # cookie = [item['name'] + "=" + item['value'] for item in cookie]
    # cookiestr = '; '.join(item for item in cookie)
    # print(cookiestr)
    # headers1 = {
    #     'Host': 'www.toutiao.com',
    #     "Accept": "*/*",
    #     "Accept-Encoding": "gzip, deflate, sdch",
    #     "Accept-Language": "zh-CN,zh;q=0.8",
    #     "Cache-Control": "no-cache",
    #     "Connection": "keep-alive",
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',
    #     'Referer': 'https://www.toutiao.com/ch/gallery_photograthy/',
    #     "Cookie": cookiestr
    # }

    data = {
        'category': keyword,
        'utm_source': 'toutiao',
        'max_behot_time': str(max_behot_time),
        'as': eas,
        'cp': ecp,
        '_signature': _signature,
    }
    url = 'https://www.toutiao.com/api/pc/feed/?' + urlencode(data)
    print("当前json数据的url：", url)

    browser.get(url)
    print("正在访问：", url)
    print('*'*20)

    bps = browser.page_source
    soup = BeautifulSoup(bps, 'lxml')
    json_text = str(soup.select('pre')[0].string)
    # print(json_text, type(json_text))
    j = json.loads(json_text)
    # print(j)

    # 时间戳，第一次为0，之后在json数据中直接提取
    max_behot_time_tmp = j['next']['max_behot_time']
    # print(max_behot_time_tmp)

    # 尝试直接用cookies获取，失败！！
    # str_page = get_page(url, headers1)
    # j = json.loads(str_page)
    # print(j)
    # max_behot_time_tmp = j['next']['max_behot_time']

    return json_text, max_behot_time_tmp


def main(keyword):
    try:
        mbt = 0
        for i in range(10):
            html, mbt_tmp = get_json_page(keyword, max_behot_time=mbt)
            # print(html, mbt_tmp)
            mbt = mbt_tmp
            for url in parse_json_page(html):
                print("当前页面地址：", url)
                if url:
                    # html = get_page_detail(url)
                    html = get_page(url, headers)
                    if html:
                        result = parse_page_detail(html, url)
                        print("结果如下：")
                        print(result)
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    try:
        while True:
            command = input("""请输入对应的数字或字母选择爬取或退出：\n
1:全部\n
2:老照片\n
3:摄影集\n
q:退出程序\n
>>""")

            if command == 'q':
                print("退出程序...")
                break
            if command == '1':
                print("将为您爬取全部图片...")
                kw = '组图'
            elif command == '2':
                print("将为您爬取老照片...")
                kw = 'gallery_old_picture'
            elif command == '3':
                print("将为您爬取摄影集...")
                kw = 'gallery_photograthy'
            else:
                print("输入有误，请确认你的输入！")
                continue
            main(kw)
    except Exception as e:
        print(e)
    finally:
        browser.quit()
