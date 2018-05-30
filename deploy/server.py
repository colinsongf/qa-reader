"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-29 11:11:31
	* @modify date 2018-05-30 01:10:48
	* @desc [server for app]
"""

import os
import sys
import jieba
import pickle
import importlib
importlib.reload(sys)
sys.path.append('..')
import tensorflow as tf
from collections import OrderedDict
from utils import get_feature
from utils import find_best_answer
from search import Search
from readers import DuBidaf
from readers import QANet
from utils import Vocab


def choose_algo(algo, vocab, config):
    """
    choose the algorithm
    """
    if algo == 'BIDAF':
        rc_model = DuBidaf(vocab, config)
    elif algo == 'QANET':
        rc_model = QANet(vocab, config)
    else:
        rc_model = None
    return rc_model


class Server(object):
    def __init__(self, args, config):
        self.algo = args.algo
        self.config = config
        self.search_engine = Search(
            config.solr_core, config.solr_url, config.baidu_url, config.limit)
        self.vocab = Vocab()
        self._init_model()
        self._init_sess()

    def _init_model(self):
        with open(os.path.join(self.config.vocab_dir,
                               self.config.dataset_name +
                               '_vocab.data'), 'rb') as fin:
            self.vocab = pickle.load(fin)

        self.model = choose_algo(self.algo, self.vocab, self.config)

    def _init_sess(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(
            self.sess,  os.path.join(self.config.model_dir, self.algo))

    def search(self, question, source):
        if source == 'solr':
            documents = self.search_engine.solr_search(question)
        elif source == 'baidu':
            documents = self.search_engine.baidu_search(question)
        elif source == 'both':
            documents1 = self.search_engine.solr_search(question)
            documents2 = self.search_engine.baidu_search(question)
            documents = documents1 + documents2
        else:
            raise NotImplementedError('{} is not support yet.'.format(source))
        return documents

    def inference(self, question, source='solr'):
        if not source:
            source = 'solr'
        documents = self.search(question, source)
        assert len(documents) != 0, 'Can not search any passages.'

        response = []
        for document in documents:
            result = OrderedDict()
            passage = document['passage']
            source = document['source']
            inps = (passage, question)
            feed_dict = get_feature(self.model, inps, self.vocab, self.config)
            start_prob, end_prob = self.sess.run(
                [self.model.start_probs, self.model.end_probs], feed_dict=feed_dict)
            best_start, best_end, max_prob = find_best_answer(
                start_prob[0], end_prob[0], self.model.max_a_len)
            answer = ''.join(list(jieba.cut(passage))[
                             best_start: best_end + 1])
            score = max_prob
            result['answer'] = answer
            result['score'] = str(score)
            result['passage'] = passage
            result['source'] = source
            response.append(result)
        response = sorted(response, key=lambda x: float(
            x['score']), reverse=True)
        return {'response': response}
