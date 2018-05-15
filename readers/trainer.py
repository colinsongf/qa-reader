"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-15 10:29:17
	* @modify date 2018-05-15 01:56:15
	* @desc [description]
"""

import tensorflow as tf
from utils import get_feed_dict, average_gradients, get_opt


class Trainer(object):
    def __init__(self, config, model):
        self.config = config
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

    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch, get_summary=False):
        feed_dict = get_feed_dict(model, batch, self.config.dropout_keep_prob)

        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op],
                         feed_dict=feed_dict)
        else:
            loss, train_op = sess.run(
                [self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


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
