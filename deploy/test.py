import bottle
from bottle import route, run
import threading
import json
import numpy as np


from time import sleep

'''
This file is taken and modified from R-Net by Minsangkim142
https://github.com/minsangkim142/R-net
'''

app = bottle.Bottle()


@app.get("/")
def home():
    with open('../data/demo2.html', 'r') as fl:
        html = fl.read()
        return html


@app.post('/answer')
def answer():
    question = bottle.request.json['question']
    print("received question: {}".format(question))
    response = {"answer": 'receive: ' + question}
    return response


def main():
    app.run(port=8080, host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main()
