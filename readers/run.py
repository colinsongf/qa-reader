"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-07 04:17:16
	* @modify date 2018-05-10 01:26:40
	* @desc [run the model system]
"""

import sys
import importlib
importlib.reload(sys)
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import json
import argparse
import logging
import tensorflow as tf
from brc_dataset import BRCDataset
from cmrc_dataset import CMRCDataset
from vocab import Vocab
from rc_model.dubidaf import DuBidaf
from rc_model.mlstm import Mlstm
from rc_model.qanet import QANet
from rc_model.rnet import Rnet
from rc_model.baseline import BaseModel
from utils import Config
from demo import Demo
from trainer import Trainer, MultiGPUTrainer
from evaluator import Evaluator


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        'Reading Comprehension')
    parser.add_argument('--algo', choices=['BIDAF', 'MLSTM', 'QANET', 'RNET', 'BASE'], default='BIDAF',
                        help='choose the algorithm to use')
    parser.add_argument('--app_prof', choices=['dureader_debug', 'cmrc2018_debug', 'dureader', 'cmrc2018'],
                        default='cmrc2018_debug',
                        help='choose config profile to use')
    parser.add_argument('--params_prof', choices=['qanet', 'default'], default='qanet',
                        help='choose params profile to use')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--demo', action='store_true',
                        help='run demo')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--restore', action='store_true',
                        help='restore the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--use_gpus', action='store_true',
                        help='run demo')
    return parser.parse_args()


def choose_algo(algo, vocab, config):
    """
    choose the algorithm
    """
    if algo == 'BIDAF':
        rc_model = DuBidaf(vocab, config)
    elif algo == 'MLSTM':
        rc_model = Mlstm(vocab, config)
    elif algo == 'QANET':
        rc_model = QANet(vocab, config)
    elif algo == 'RNET':
        rc_model = Rnet(vocab, config)
    elif algo == 'BASE':
        rc_model = BaseModel(vocab, config)
    else:
        rc_model = None
    return rc_model


def get_multi_gpu_models(algo, vocab, config):
    models = []
    for gpu_idx in range(config.num_gpu):
        with tf.name_scope('model_{}'.format(gpu_idx)) as scope, \
                tf.device("/gpu:{}".format(gpu_idx)):
            model = choose_algo(algo, vocab, config)
            tf.get_variable_scope().reuse_variables()
            models.append(model)
    return models


def prepare(config):
    """
    checks data, creates the directories, 
    prepare the vocabulary and embeddings
    """
    logger = logging.getLogger('qarc')
    logger.info('Checking the data files...')
    for data_path in config.train_files + config.dev_files + config.test_files:
        assert os.path.exists(data_path),\
            '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    train_summary_dir = os.path.join(config.summary_dir, 'train')
    dev_summary_dir = os.path.join(config.summary_dir, 'dev')
    for dir_path in [config.vocab_dir, config.model_dir, config.result_dir, train_summary_dir, dev_summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Load dataset...')
    if config.dataset_name.startswith('cmrc2018'):
        qarc_data = CMRCDataset(config.max_p_len, config.max_q_len, config.max_char_len, config.max_py_len,
                                config.train_files, config.dev_files, config.test_files)
    else:
        qarc_data = BRCDataset(config.max_p_num, config.max_p_len, config.max_q_len, config.max_char_len,
                               config.train_files, config.dev_files, config.test_files)

    logger.info('Building vocabulary...')
    vocab = Vocab(lower=True)
    for word in qarc_data.word_iter('train'):
        vocab.add_word(word)
    for char in qarc_data.char_iter('train'):
        vocab.add_char(char)
    for py in qarc_data.py_iter('train'):
        vocab.add_py(py)

    unfiltered_vocab_word_size = vocab.word_size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_word_num = unfiltered_vocab_word_size - vocab.word_size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(
        filtered_word_num, vocab.word_size()))

    unfiltered_vocab_char_size = vocab.char_size()
    vocab.filter_chars_by_cnt(min_cnt=2)
    filtered_char_num = unfiltered_vocab_char_size - vocab.char_size()
    logger.info('After filter {} chars, the final chars size is {}'.format(
        filtered_char_num, vocab.char_size()))

    unfiltered_vocab_py_size = vocab.py_size()
    vocab.filter_pys_by_cnt(min_cnt=2)
    filtered_py_num = unfiltered_vocab_py_size - vocab.py_size()
    logger.info('After filter {} pys, the final pys size is {}'.format(
        filtered_py_num, vocab.py_size()))

    logger.info('Assigning word embeddings...')
    vocab.load_pretrained_word_embeddings(
        config.word2vec, config.word_embed_dim)

    logger.info('Assigning char embeddings...')
    # vocab.randomly_init_char_embeddings(config.char_embed_dim)
    vocab.load_pretrained_char_embeddings(
        config.word2vec, config.char_embed_dim)

    logger.info('Assigning py embeddings...')
    vocab.randomly_init_py_embeddings(config.py_embed_dim)

    logger.info('Saving vocab...')
    with open(os.path.join(config.vocab_dir, config.dataset_name + '_vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args, config):
    """
    trains the rc model
    """
    logger = logging.getLogger('qarc')
    logger.info('Load dataset and vocab...')
    with open(os.path.join(config.vocab_dir, config.dataset_name + '_vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    if config.dataset_name.startswith('cmrc2018'):
        qarc_data = CMRCDataset(config.max_p_len, config.max_q_len, config.max_char_len, config.max_py_len,
                                config.train_files, config.dev_files, config.test_files)
    else:
        qarc_data = BRCDataset(config.max_p_num, config.max_p_len, config.max_q_len, config.max_char_len,
                               config.train_files, config.dev_files, config.test_files)
    logger.info('Converting text into ids...')
    qarc_data.convert_to_ids(vocab)

    if args.use_gpus:
        logger.info('Init multi gpu trainer...')
        # rc_models = get_multi_gpu_models(args.algo, vocab, config)
        trainer = MultiGPUTrainer(config, vocab, args.algo)
    else:
        logger.info('Init single gpu trainer...')
        rc_model = choose_algo(args.algo, vocab, config)
        trainer = Trainer(config, rc_model, vocab)
    if args.restore:
        logger.info('Restoring the model...')
        trainer.restore(model_dir=config.model_dir, model_prefix=args.algo)
    else:
        logger.info('Initialize the model...')

    logger.info('Training the model...')
    trainer.train(qarc_data, config.epochs, config.batch_size,
                  save_dir=config.model_dir,
                  save_prefix=args.algo)
    logger.info('Done with model training!')


def evaluate(args, config):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger('qarc')
    logger.info('Load data_set and vocab...')
    with open(os.path.join(config.vocab_dir, config.dataset_name + '_vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(config.dev_files) > 0, 'No dev files are provided.'
    if config.dataset_name.startswith('cmrc2018'):
        qarc_data = CMRCDataset(config.max_p_len, config.max_q_len, config.max_char_len,
                                config.train_files, config.dev_files, config.test_files)
    else:
        qarc_data = BRCDataset(config.max_p_num, config.max_p_len, config.max_q_len, config.max_char_len,
                               config.train_files, config.dev_files, config.test_files)
    logger.info('Converting text into ids...')
    qarc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = choose_algo(args.algo, vocab, config)
    if not rc_model:
        raise NotImplementedError(
            'The algorithm {} is not implemented.'.format(args.algo))
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    evaluator = Evaluator(
        config, rc_model, sess, saver)
    evaluator.restore(model_dir=config.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = qarc_data.gen_mini_batches('dev', config.batch_size,
                                             pad_id=vocab.get_word_id(vocab.pad_token), shuffle=False)
    dev_loss, f1, em, avg = evaluator.evaluate(
        dev_batches, result_dir=config.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info(
        'Dev eval result: F1: {:.3f} EM: {:.3f} AVG: {:.3f}'.format(f1, em, avg))
    logger.info('Predicted answers are saved to {}'.format(
        os.path.join(config.result_dir)))


def predict(args, config):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("qarc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(config.vocab_dir, config.dataset_name + '_vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(config.test_files) > 0, 'No test files are provided.'

    if config.dataset_name.startswith('cmrc2018'):
        qarc_data = CMRCDataset(config.max_p_len, config.max_q_len, config.max_char_len,
                                config.train_files, config.dev_files, config.test_files)
    else:
        qarc_data = BRCDataset(config.max_p_num, config.max_p_len, config.max_q_len, config.max_char_len,
                               config.train_files, config.dev_files, config.test_files)
    logger.info('Converting text into ids...')
    qarc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = choose_algo(args.algo, vocab, config)
    if not rc_model:
        raise NotImplementedError(
            'The algorithm {} is not implemented.'.format(args.algo))
    # saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    evaluator = Evaluator(
        config, rc_model, sess, saver)
    evaluator.restore(model_dir=config.model_dir, model_prefix=args.algo)
    logger.info('Test the model on test set...')
    test_batches = qarc_data.gen_mini_batches('test', config.batch_size,
                                              pad_id=vocab.get_word_id(vocab.pad_token), shuffle=False)
    evaluator.predict(
        test_batches, result_dir=config.result_dir, result_prefix='test.predicted')


def demo(args, config):
    logger = logging.getLogger('qarc')
    logger.info('Start demo...')

    logger.info('Load vocab...')
    with open(os.path.join(config.vocab_dir, config.dataset_name + '_vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    model = choose_algo(args.algo, vocab, config)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(
        sess,  os.path.join(config.model_dir, args.algo))

    demo = Demo(sess, config, model)


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    dic = {'../data/configs.yaml': args.app_prof,
           '../data/params.yaml': args.params_prof}
    config = Config(dic)

    logger = logging.getLogger("qarc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info('Running with args : {}'.format(config))

    if args.prepare:
        prepare(config)
    if args.train:
        train(args, config)
    if args.evaluate:
        evaluate(args, config)
    if args.predict:
        predict(args, config)
    if args.demo:
        demo(args, config)


if __name__ == '__main__':
    run()
