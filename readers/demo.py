#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import bottle
from bottle import route, run
import threading
import json
import numpy as np
import jieba


from time import sleep

'''
This file is taken and modified from R-Net by Minsangkim142
https://github.com/minsangkim142/R-net
'''

app = bottle.Bottle()
query = []
response = ""
score = ""


@app.get("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html


@app.post('/answer')
def answer():
    passage = bottle.request.json['passage']
    question = bottle.request.json['question']
    print("received question: {}".format(question))
    # if not passage or not question:
    #     exit()
    global query, response, score
    query = (passage, question)
    while not response:
        sleep(0.1)
    print("received response: {}  score: {}".format(response, score))
    response_ = {"answer": response, "score": score}
    response = []
    return response_


def find_best_answer(start_prob, end_prob, max_a_len):

    passage_len = len(start_prob)
    best_start, best_end, max_prob = -1, -1, 0
    for start_idx in range(passage_len):
        for ans_len in range(max_a_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_prob[start_idx] * end_prob[end_idx]
            if prob > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob
    return best_start, best_end, max_prob


class Demo(object):
    def __init__(self, sess, config, model):
        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.demo_backend, args=[
                         sess, config, model, run_event]).start()
        app.run(port=8080, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()

    def demo_backend(self, sess, config, model, run_event):
        global query, response, score

        while run_event.is_set():
            sleep(0.1)
            if query:
                passage_tokens = list(jieba.cut(query[0]))
                question_tokens = list(jieba.cut(query[1]))
                passage_token_ids = model.vocab.convert_word_to_ids(
                    passage_tokens)
                question_token_ids = model.vocab.convert_word_to_ids(
                    question_tokens)
                feed_dict = {model.p: [passage_token_ids],
                             model.q: [question_token_ids],
                             model.p_length: [len(passage_tokens)],
                             model.q_length: [len(question_tokens)],
                             model.start_label: [0],
                             model.end_label: [0],
                             model.dropout: config.dropout}

                start_prob, end_prob = sess.run(
                    [model.start_probs, model.end_probs], feed_dict=feed_dict)
                # print(query)
                best_start, best_end, max_prob = find_best_answer(
                    start_prob[0], end_prob[0], model.max_a_len)
                response = ''.join(
                    passage_tokens[best_start:best_end + 1])
                score = str(max_prob)
                query = []
