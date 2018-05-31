"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-24 10:31:32
	* @modify date 2018-05-24 10:31:32
	* @desc [implements the search interface]
"""
from utils import SolrSearch
from utils import BaiduSearch


class Search(object):
    def __init__(self, solr_core, solr_url, baidu_url, limit):
        self.solr = SolrSearch(solr_core, solr_url, limit)
        self.baidu = BaiduSearch(baidu_url, limit)

    def baidu_search(self, question):
        passages = self.baidu.query(question)
        result = [{'passage': passage, 'source': 'baidu'}
                  for passage in passages]
        return result

    def solr_search(self, question):
        passages = self.solr.query(question)
        result = [{'passage': passage, 'source': 'solr'}
                  for passage in passages]
        return result

    def search(self, question):
        pass
