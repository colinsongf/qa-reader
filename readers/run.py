"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-07 04:17:16
	* @modify date 2018-05-07 04:17:16
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
from dataset import Dataset
from vocab import Vocab
from rc_model.bidaf import Bidaf
from rc_model.mlstm import Mlstm
from rc_model.qanet import QAnet
from rc_model.rnet import Rnet
from utils import Config


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        'Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--algo', choices=['BIDAF', 'MLSTM', 'QANET', 'RNET'], default='BIDAF',
                        help='choose the algorithm to use')
    parser.add_argument('--app_prof', choices=['dureader_debug', 'cmrc2018_debug', 'dureader', 'cmrc2018'],
                        default='cmrc2018_debug',
                        help='choose config profile to use')
    parser.add_argument('--params_prof', choices=['gru', 'lstm'], default='lstm',
                        help='choose params profile to use')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--restore', action='store_true',
                        help='restore the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    return parser.parse_args()


def choose_algo(algo, vocab, config):
    """
    choose the algorithm
    """
    if algo == 'BIDAF':
        rc_model = Bidaf(vocab, config)
    elif algo == 'MLSTM':
        rc_model = Mlstm(vocab, config)
    elif algo == 'QANET':
        rc_model = QAnet(vocab, config)
    elif algo == 'RNET':
        rc_model = Rnet(vocab, config)
    else:
        rc_model = None
    return rc_model


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
    for dir_path in [config.vocab_dir, config.model_dir, config.result_dir, config.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Load dataset...')
    qarc_data = Dataset(config.max_p_num, config.max_p_len, config.max_q_len,
                        config.train_files, config.dev_files, config.test_files)

    logger.info('Building vocabulary...')
    vocab = Vocab(lower=True)
    for word in qarc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(
        filtered_num, vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.load_pretrained_embeddings(config.word2vec, config.embed_size)

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

    qarc_data = Dataset(config.max_p_num, config.max_p_len, config.max_q_len,
                        config.train_files, config.dev_files)
    logger.info('Converting text into ids...')
    qarc_data.convert_to_ids(vocab)

    rc_model = choose_algo(args.algo, vocab, config)
    if not rc_model:
        raise NotImplementedError(
            'The algorithm {} is not implemented.'.format(args.algo))
    if args.restore:
        logger.info('Restoring the model...')
        rc_model.restore(model_dir=config.model_dir, model_prefix=args.algo)
    else:
        logger.info('Initialize the model...')

    logger.info('Training the model...')
    rc_model.train(qarc_data, config.epochs, config.batch_size,
                   save_dir=config.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=config.dropout_keep_prob)
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
    qarc_data = Dataset(config.max_p_num, config.max_p_len, config.max_q_len,
                        config.train_files, config.dev_files)
    logger.info('Converting text into ids...')
    qarc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = choose_algo(args.algo, vocab, config)
    if not rc_model:
        raise NotImplementedError(
            'The algorithm {} is not implemented.'.format(args.algo))
    rc_model.restore(model_dir=config.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = qarc_data.gen_mini_batches('dev', config.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=config.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
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

    qarc_data = BRCDataset(config.max_p_num, config.max_p_len, config.max_q_len,
                           test_files=config.test_files)
    logger.info('Converting text into ids...')
    qarc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = choose_algo(args.algo, vocab, config)
    if not rc_model:
        raise NotImplementedError(
            'The algorithm {} is not implemented.'.format(args.algo))
    rc_model.restore(model_dir=config.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = qarc_data.gen_mini_batches('test', config.batch_size,
                                              pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=config.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    dic = {'configs.yaml': args.app_prof, 'params.yaml': args.params_prof}
    config = Config(dic)
    print(config)


def main():
    run()


if __name__ == '__main__':
    main()