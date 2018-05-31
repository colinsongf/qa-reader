"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-30 01:15:50
	* @modify date 2018-05-30 01:15:50
	* @desc [utils for solr search]
"""

import requests
from SolrClient import SolrClient
from bs4 import BeautifulSoup
from lxml import etree


class SolrSearch(object):
    def __init__(self, core, url, limit=3):
        self.core = core
        self.url = url
        self.limit = limit
        self.solr = SolrClient(url)

    def query(self, question):
        passages = set()
        res = self.solr.query(self.core, {
            'q': 'context_text:{}'.format(question)
        })
        for doc in res.docs:
            passages.add(doc['context_text'][0])
        return list(passages)[:self.limit]


class BaiduSearch(object):
    def __init__(self, url, limit=2):
        self.url = url
        self.limit = limit

    def query(self, question):
        herfs = self.get_herf(question)
        passages = self.get_passage(herfs)
        return passages

    def get_herf(self, question):
        response = requests.get(self.url + 'word={}'.format(question))
        soup = BeautifulSoup(response.content, 'lxml')
        tis = soup.find_all(class_='ti')
        hrefs = [ti.attrs['href'] for ti in list(tis)][:self.limit]
        # print(hrefs)
        return hrefs

    def get_passage(self, hrefs):
        passages = list()
        for href in hrefs:
            response = requests.get(href)
            soup = BeautifulSoup(response.content, 'lxml')
            best_text = soup.find(class_='best-text mb-10')
            if best_text:
                passage = best_text.text
                passages.append(passage)
            else:
                continue
        return passages


def main():
    solr = SolrSearch('cmrc2018_core', 'http://10.89.100.14:8999/solr')
    passages = solr.query('泡泡战士中, 生命数耗完即算为什么？')
    # baidu = BaiduSearch(
    #     'https://zhidao.baidu.com/search?')
    # passages = baidu.query('姚明多高')
    print(passages)


if __name__ == '__main__':
    main()
