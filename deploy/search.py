"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-24 10:31:32
	* @modify date 2018-05-24 10:31:32
	* @desc [implements the search interface]
"""


class Search(object):
    def __init__(self):
        pass

    def baidu_search(self, question):
        return [{'passage': '这是一个示例', 'source': 'baidu'}]

    def solr_search(self, question):
        return [{'passage': '这还是一个示例哇哈哈', 'source': 'solr'}]

    def search(self, question):
        pass
