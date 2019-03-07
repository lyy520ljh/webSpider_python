# -------------------------------------------------------------------------
#   程序：download_music.py
#   版本：0.1
#   作者：lyy
#   日期：编写日期2018/12/20
#   语言：Python 3.6.x
#   操作：python ownload_music.py
#   功能：	根据输入的歌曲/歌手下载QQ音乐
#
# -------------------------------------------------------------------------

from urllib.parse import quote
from fake_useragent import UserAgent
import requests
import time
import os


ua = UserAgent()
# print(headers)


def get_keyword():
    keyword = input('请输入歌曲/歌手,若有多首/个请以中文逗号"，"隔开：')
    keywordlist = keyword.split('，')
    return keywordlist


def get_purl(url):
    headers = {'user-agent': ua.random}
    response2 = requests.get(url=url, headers=headers)
    jsonfile2 = response2.json()
    # print(jsonfile2)
    purl = jsonfile2['req_0']['data']['midurlinfo'][0]['purl']
    return purl


def download_music(album_name, title, singer, purl):
    headers = {'user-agent': ua.random}
    download_url = 'http://dl.stream.qqmusic.qq.com/%s' % purl
    # print(download_url)
    response = requests.get(download_url, headers=headers)
    if not os.path.exists('music/' + title):
        os.mkdir('music/' + title)
    with open('music/' + title + '/' + "专辑：" + album_name + ',' + "歌曲：" + title + ',' + "歌手：" + singer + '.mp3', 'wb') as f:
        f.write(response.content)
    print("歌曲下载完成")


def get_music_detail(keyword):
    headers = {'user-agent': ua.random}
    url1 = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp?ct=24&qqmusic_ver=1298&new_json=1&remoteplace=txt.yqq.song&t=0&aggr=1&cr=1&catZhida=1&lossless=0&flag_qc=0&p=1&n=20&w=%s&g_tk=5381&loginUin=0&hostUin=0&format=json&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq.json&needNewCode=0' % keyword

    response1 = requests.get(url=url1, headers=headers)
    jsonfile = response1.json()
    songlist = jsonfile['data']['song']['list']
    i = 0
    for song in songlist:
        # print(songfile['file']['media_mid'])
        # media_mid = song['file']['media_mid']
        media_mid = song['mid']
        album_name = song['album']['name']
        title = song['title']
        singer = '+'.join([singer['name'] for singer in song['singer']])

        print('-' * 30)
        print(media_mid, album_name, title, singer, end='\n')
        print('-' * 30)

        time.sleep(3)

        data = '{"req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey","param":{"guid":"1308200071","songmid":["%s"],"songtype":[0],"uin":"0","loginflag":1,"platform":"20"}},"comm":{"uin":0,"format":"json","ct":20,"cv":0}}' % media_mid
        url2 = 'https://u.y.qq.com/cgi-bin/musicu.fcg?&data={}'.format(quote(data))
        # print(url2)
        purl = get_purl(url=url2)
        download_music(album_name, title, singer, purl)
        i += 1

        # 每首歌只下载排行最前的两个版本
        if i >= 2:
            break


def main():
    for keyword in get_keyword():
        get_music_detail(keyword)


if __name__ == "__main__":
    main()
