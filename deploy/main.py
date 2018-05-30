"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-23 09:13:22
	* @modify date 2018-05-23 03:44:17
	* @desc [run the interface of qa system]
"""

import argparse
import logging


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('interface')
    parser.add_argument('--algo', choices=['BIDAF', 'MLSTM', 'QANET', 'RNET'], default='BIDAF',
                        help='choose the algorithm to use')
    parser.add_argument('--source', choices=['solr', 'baidu'], default='solr',
                        help='choose the algorithm to use')
    parser.add_argument('--app_prof', choices=['dureader_debug', 'cmrc2018_debug', 'dureader', 'cmrc2018'],
                        default='cmrc2018',
                        help='choose config profile to use')
    parser.add_argument('--params_prof', choices=['qanet', 'default'], default='qanet',
                        help='choose params profile to use')

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
    else:
        rc_model = None
    return rc_model


def interface(args, config):
    logger = logging.getLogger('qarc')
    logger.info('Start interface...')

    logger.info('Load vocab...')
    with open(os.path.join(config.vocab_dir, config.dataset_name + '_vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    model = choose_algo(args.algo, vocab, config)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(
        sess,  os.path.join(config.model_dir, args.algo))

    interface = Interface(sess, config, model)


def main():
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


if __name__ == '__main__':
    main()
