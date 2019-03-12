# -------------------------------------------------------------------------
#   程序：taobao_selenium.py
#   版本：0.1纪念版
#   作者：lyy
#   日期：编写日期2018/09/11
#   语言：Python 3.6.x
#   操作：python taobao_selenium.py
#   功能：抓取淘宝美食,保存到json文件中
#   备注：版本已过期，selenium会被检测到，跳转到登录界面
#
# -------------------------------------------------------------------------
import re
import json
from pyquery import PyQuery as pq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC

browser = webdriver.Chrome()
wait = WebDriverWait(browser, 20)


def search(keyword):
    try:
        browser.get("https://www.taobao.com")
        input_block = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#q')))
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#J_TSearchForm > div.search-button > button')))
        input_block.send_keys(keyword)
        submit.click()
        total = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsrp-pager > div > div > div > div.total')))
        get_products()
        return total.text
    except TimeoutException:
        return search(keyword)


def next_page(page_num):
    try:
        input_block = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsrp-pager > div > div > div > div.form > input')))
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#mainsrp-pager > div > div > div > div.form > span.btn.J_Submit')))
        input_block.clear()
        input_block.send_keys(page_num)
        submit.click()
        wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR,'#mainsrp-pager > div > div > div > ul > li.item.active > span'),str(page_num)))
        get_products()
    except TimeoutException:
        next_page(page_num)


def get_products():
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsrp-itemlist .items .item')))
    html = browser.page_source
    doc = pq(html)
    items = doc('#mainsrp-itemlist .items .item').items()
    for item in items:
        products = {
            'image': item.find('.pic img').attr('src'),
            'price': item.find('.price').text(),
            'deal': item.find('.deal-cnt').text()[:-3],
            'title': item.find('.title').text(),
            'shop': item.find('.shop').text(),
            'location': item.find('.location').text()
        }
        # print(products)
        json_str = json.dumps(products, ensure_ascii=False)
        with open('products.json', 'a') as f:
            f.write(json_str + '\n')


def main():
    keyword = input("请输入要抓取的相关商品：")
    total = search(keyword)
    total = int(re.compile('(\d+)').search(total).group(1))
    # print(total)
    for i in range(2, total+1):
        next_page(i)


if __name__ == '__main__':
    main()
