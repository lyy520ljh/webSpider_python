# -------------------------------------------------------------------------
#   程序：tiobe.py
#   版本：0.1
#   作者：lyy
#   日期：编写日期2018/09/11
#   语言：Python 3.6.x
#   操作：python tiobe.py
#   功能：抓取TIOBE指数前20名排行开发语言,保存到mysql
#
# -------------------------------------------------------------------------

import pymysql
import requests
from lxml import etree
from lxml.etree import ParseError
from requests.exceptions import RequestException

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"}


def get_page(url):
    text = None
    try:
        response = requests.get(url, headers=headers)
        text = response.text
    except RequestException as e:
        print("requests is error!", e)

    try:
        html = etree.HTML(text, etree.HTMLParser())
        result = html.xpath('//table[contains(@class,"top20")]/tbody/tr//text()')
        position = 0
        for i in range(20):
            yield result[position:position + 5]
            position += 5
    except ParseError as e:
        print(e)


def save_data_to_mysql(data):
    db = pymysql.connect('localhost', 'root', '******', 'spider')
    cursor = db.cursor()
    print(cursor)
    sql = 'insert into tiobe (n2019, n2018, programming_Language, Rating, Change1) values ("%d", "%d", "%s", "%s", "%s")'
    for d in data:
        # print(d)
        try:
            # s = sql % (int(d[0]), int(d[1]), d[2], d[3], d[4])
            # print(s)
            cursor.execute(sql % (int(d[0]), int(d[1]), d[2], d[3], d[4]))
            db.commit()
        except Exception as e:
            print(e)
            db.rollback()
    db.close()


def main():
    url = 'https://www.tiobe.com/tiobe-index/'
    data = get_page(url)
    save_data_to_mysql(data)
    print('ok')


if __name__ == "__main__":
    main()
