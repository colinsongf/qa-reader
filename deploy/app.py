import os
import sys
import logging
import argparse
import threading
from flask import Flask, request, jsonify
from time import sleep
from server import Server
from utils import Config


parser = argparse.ArgumentParser('interface')
parser.add_argument('--algo', choices=['BIDAF', 'MLSTM', 'QANET', 'RNET'], default='BIDAF',
                    help='choose the algorithm to use')
parser.add_argument('--source', choices=['solr', 'baidu'], default='solr',
                    help='choose the algorithm to use')
parser.add_argument('--app_prof', choices=['dureader_debug', 'cmrc2018_debug', 'dureader', 'cmrc2018'],
                    default='cmrc2018_debug',
                    help='choose config profile to use')
parser.add_argument('--params_prof', choices=['qanet', 'default'], default='qanet',
                    help='choose params profile to use')
args = parser.parse_args()


dic = {'../data/configs.yaml': args.app_prof,
       '../data/params.yaml': args.params_prof}
config = Config(dic)

server = Server(args, config)


app = Flask(__name__)


@app.route('/')
def index():
    return 'Index Page'


@app.route('/answer', methods=['GET', 'POST'])
def hello():
    args = request.args
    question = args.get('question', None)
    source = args.get('source', None)
    result = {'question': None, 'response': None}
    if not question:
        return jsonify(result)
    result['question'] = question
    response = server.inference(question=question, source='solr')
    result.update(response)
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    # main()
    app.run(debug=True)
