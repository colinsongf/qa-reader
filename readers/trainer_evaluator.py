"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-15 10:29:17
	* @modify date 2018-05-15 01:56:15
	* @desc [description]
"""

import logging
import tensorflow as tf
from utils import get_feed_dict, average_gradients, get_opt


class Trainer(object):
    def __init__(self, sess, config, model, vocab):
        self.sess = sess
        self.config = config
        self.vocab = vocab
        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.model = model
        self.opt = get_opt(self.optim_type, self.learning_rate)
        self.loss = model.get_loss()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.grads = self.opt.compute_gradients(self.loss)
        self.train_op = self.opt.apply_gradients(
            self.grads, global_step=self.global_step)
        self.evaluator = Evaluator()
        self.logger = logging.getLogger('qarc')
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def get_train_op(self):
        return self.train_op

    def step(self, batch, get_summary=False):
        feed_dict = get_feed_dict(model, batch, self.config.dropout_keep_prob)

        if get_summary:
            loss, summary, train_op = \
                self.sess.run([self.loss, self.summary, self.train_op],
                              feed_dict=feed_dict)
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
        for bitx, batch in enumerate(train_batches, 1):
            loss, summary, train_op = self.step(batch)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
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
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches(
                'train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info(
                'Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info(
                    'Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches(
                        'dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluator.evaluate(
                        eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
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


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        self.model = model
        self.config = config
        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.model = model
        self.opt = get_opt(self.optim_type, self.learning_rate)
        self.global_step = model.get_global_step()
        self.summary = model.summary
        # self.models = models
        losses = []
        grads_list = []
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/gpu:{}".format(gpu_idx)):
                loss = model.get_loss()
                grads = self.opt.compute_gradients(loss)
                losses.append(loss)
                grads_list.append(grads)
        self.loss = tf.add_n(losses) / len(losses)
        self.grads = average_gradients(grads_list)
        self.train_op = self.opt.apply_gradients(
            self.grads, global_step=self.global_step)

    def step(self, sess, batches, get_summary=False):
        pass


class Evaluator(object):
    pass
