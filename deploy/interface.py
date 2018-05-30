"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-24 09:51:37
	* @modify date 2018-05-24 09:51:37
	* @desc [interface]
"""

import sys
import importlib
importlib.reload(sys)
sys.path.append('..')
import json
import jieba
import bottle
import threading
import numpy as np
import tensorflow as tf
from bottle import route, run
from time import sleep
from search import Search
from utils import get_feature
from utils import find_best_answer

app = bottle.Bottle()
search = Search()
query = tuple()
response = list()


@app.get("/")
def home():
    with open('../data/interface.html', 'r') as fl:
        html = fl.read()
        return html


@app.post('/answer')
def answer():
    question = bottle.request.json['question']
    passages = search.search(question)
    print("received question: {}".format(question))
    # if not passage or not question:
    #     exit()
    global query, response
    query = (passages, question)
    while not response:
        sleep(0.1)
    print("received response: {}".format(response))
    response_ = response
    response = list()
    return response_


class Interface(object):
    def __init__(self, sess, model, config):
        self.model = model
        self.vocab = vocab
        self.config = config

        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.backend, args=[
                         sess, model, config, run_event]).start()
        app.run(port=8080, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()

    def backend(self, sess, model, config, run_event):
        global query, response

        while run_event.is_set():
            sleep(0.1)
            if query:
                local_response = list()
                for passage in query[0]:
                    inp = (passage, query[1])
                    passage_tokens, passage_token_ids, question_token_ids, \
                        p_length, q_length, passage_char_ids, \
                        question_char_ids = get_feature(
                            inp, model.vocab, config)
                    feed_dict = {model.p: [passage_token_ids],
                                 model.q: [question_token_ids],
                                 model.p_length: [p_length],
                                 model.q_length: [q_length],
                                 model.ph: [passage_char_ids],
                                 model.qh: [question_char_ids],
                                 model.start_label: [0],
                                 model.end_label: [0],
                                 model.dropout: config.dropout}
                    start_prob, end_prob = sess.run(
                        [model.start_probs, model.end_probs], feed_dict=feed_dict)

                    best_start, best_end, max_prob = find_best_answer(
                        start_prob[0], end_prob[0], model.max_a_len)

                    answer = ''.join(
                        passage_tokens[best_start: best_end + 1])
                    score = str(max_prob)

                    local_response.append({'answer': answer, 'score': score})
                response = local_response
                local_response = list()
                query = tuple()
