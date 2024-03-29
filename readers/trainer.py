"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-15 10:29:17
	* @modify date 2018-05-15 01:56:15
	* @desc [description]
"""

import os
import logging
import tensorflow as tf
from evaluator import Evaluator
from rc_model.dubidaf import DuBidaf
from rc_model.mlstm import Mlstm
from rc_model.qanet import QANet
from rc_model.rnet import Rnet
from general import get_feed_dict, average_gradients, get_opt


class Trainer(object):
    def __init__(self, config, model, vocab):
        self.config = config
        self.vocab = vocab
        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.model = model
        self.opt = get_opt(self.optim_type, self.learning_rate)
        self.loss = model.get_loss()
        self.global_step = model.global_step
        self.grads = self.opt.compute_gradients(self.loss)
        self.train_op = self.opt.apply_gradients(
            self.grads, global_step=self.global_step)
        self.summary_op = None
        self.start_logits = model.start_logits
        self.end_logits = model.end_logits
        self.start_probs = model.start_logits
        self.end_probs = model.end_logits
        self.logger = logging.getLogger('qarc')
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'train'),
                                                  self.sess.graph)
        self.dev_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'dev'),
                                                self.sess.graph)
        self.evaluator = Evaluator(
            self.config, self.model, self.sess, self.saver)
        self.local_global_step = 1

    def get_train_op(self):
        return self.train_op

    def step(self, batch, get_summary=False):
        feed_dict = get_feed_dict(
            self.model, batch)
        # print(feed_dict)
        if get_summary:
            loss, summary, train_op = \
                self.sess.run([self.loss, self.summary_op, self.train_op],
                              feed_dict=feed_dict)
            # print(start_loss.shape)
        else:
            loss, train_op, start_logits, end_logits, start_probs, end_probs = self.sess.run(
                [self.loss, self.train_op, self.start_logits, self.end_logits, self.start_probs, self.end_probs], feed_dict=feed_dict)
            summary = None
            # print('1:', start_logits.shape)
            # print('2:', start_probs.shape)
            # print('3:', start_probs.shape)
            # print('4:', start_probs.shape)
        return loss, summary, train_op

    def _train_epoch(self, train_batches):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            loss, summary, train_op = self.step(batch)
            if summary:
                self.train_writer.add_summary(summary, self.local_global_step)
                self.local_global_step += 1
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_word_id(self.vocab.pad_token)
        max_avg = 0
        # global_step = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches(
                'train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches)
            self.logger.info(
                'Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info(
                    'Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches(
                        'dev', batch_size, pad_id, shuffle=False)
                    eval_loss, f1, em, avg = self.evaluator.evaluate(
                        eval_batches, result_dir=self.config.result_dir, result_prefix='dev.predicted')
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info(
                        'Dev eval result: F1: {:.3f} EM: {:.3f} AVG: {:.3f}'.format(f1, em, avg))

                    if avg > max_avg:
                        self.save(save_dir, save_prefix)
                        max_avg = avg
                else:
                    self.logger.warning(
                        'No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(
            model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(
            model_dir, model_prefix))


def choose_algo(algo, config, vocab):
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


class MultiGPUTrainer(object):

    def __init__(self, config, vocab, algo):
        self.config = config
        self.vocab = vocab
        self.models = []
        self.config = config
        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.logger = logging.getLogger('qarc')

        # with tf.variable_scope("optimizer") as scope:
        self.opt = get_opt(self.optim_type, self.learning_rate)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            losses = []
            grads_list = []
            for gpu_idx in range(config.num_gpu):
                with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/gpu:{}".format(gpu_idx)):
                    model = choose_algo(algo, config, vocab)
                    self.models.append(model)
                    loss = model.get_loss()
                    grads = self.opt.compute_gradients(loss)
                    losses.append(loss)
                    grads_list.append(grads)
                    tf.get_variable_scope().reuse_variables()
        self.global_step = self.models[0].global_step
        self.summary_op = self.models[0].summary_op
        self.loss = tf.add_n(losses) / len(losses)
        self.grads = average_gradients(grads_list)
        self.train_op = self.opt.apply_gradients(
            self.grads, global_step=self.global_step)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'train'),
                                                  self.sess.graph)
        self.dev_writer = tf.summary.FileWriter(os.path.join(config.summary_dir, 'dev'),
                                                self.sess.graph)
        self.evaluator = Evaluator(
            self.config, self.models[0], self.sess, self.saver)
        self.local_global_step = 1

    def get_train_op(self):
        return self.train_op

    def step(self, batches, get_summary=True):
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            feed_dict.update(get_feed_dict(
                model, batch))
        # print(feed_dict)
        if get_summary:
            loss, summary, train_op = \
                self.sess.run([self.loss, self.summary_op, self.train_op],
                              feed_dict=feed_dict)
            # print(start_loss.shape)
        else:
            loss, train_op = self.sess.run(
                [self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op

    def _train_epoch(self, train_batches):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        multi_batches = []
        for bitx, batch in enumerate(train_batches, 1):
            multi_batches.append(batch)
            if bitx % len(self.models) != 0:
                continue
            loss, summary, train_op = self.step(multi_batches)
            self.train_writer.add_summary(summary, self.local_global_step)
            self.local_global_step += 1
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
            multi_batches = []
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_word_id(self.vocab.pad_token)
        max_avg = 0
        # global_step = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches(
                'train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches)
            self.logger.info(
                'Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info(
                    'Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches(
                        'dev', batch_size, pad_id, shuffle=False)
                    eval_loss, f1, em, avg = self.evaluator.evaluate(
                        eval_batches, result_dir=self.config.result_dir, result_prefix='dev.predicted')
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info(
                        'Dev eval result: F1: {:.3f} EM: {:.3f} AVG: {:.3f}'.format(f1, em, avg))

                    if avg > max_avg:
                        self.save(save_dir, save_prefix)
                        max_avg = avg
                else:
                    self.logger.warning(
                        'No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(
            model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(
            model_dir, model_prefix))
