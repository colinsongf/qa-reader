"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-07 01:36:38
	* @modify date 2018-05-07 01:36:38
	* @desc [implement the BiDAF algorithm described in https://arxiv.org/abs/1611.01603]
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from rc_model.layers.basic_rnn import rnn
from rc_model.layers.match_layer import MatchLSTMLayer
from rc_model.layers.match_layer import AttentionFlowMatchLayer
from rc_model.layers.pointer_net import PointerNetDecoder


class DuBidaf(object):
    def __init__(self, vocab, config):

        # logging
        self.logger = logging.getLogger('qarc')

        # basic config
        self.hidden_size = config.hidden_size
        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.use_dropout = config.dropout_keep_prob < 1

        # length limit
        self.max_p_num = config.max_p_num
        self.max_p_len = config.max_p_len
        self.max_q_len = config.max_q_len
        self.max_a_len = config.max_a_len

        # the vocab
        self.vocab = vocab
        # self.graph = tf.Graph()
        # with self.graph.as_default():

        # self.train = train

        self._build_graph()

        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        # self._create_train_op()
        self.logger.info(
            'Time to build graph: {} s'.format(time.time() - start_t))
        # param_num = sum([np.prod(tf.Session.run(tf.shape(v)))
        #                  for v in self.all_params])
        # self.logger.info(
        #     'There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout = tf.placeholder(tf.float32)
        self.dropout_keep_prob = 1.0 - 0.5 * self.dropout

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """

        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.word_size(), self.vocab.word_embed_dim),
                initializer=tf.constant_initializer(
                    self.vocab.word_embeddings),
                trainable=False
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn(
                'bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn(
                'bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(
                self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(
                self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """

        match_layer = AttentionFlowMatchLayer(self.hidden_size)

        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(
                self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(
                    self.fuse_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            self.concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            self.no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)
                 [1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(self.concat_passage_encodes,
                                                          self.no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses, labels

        self.start_loss, self.labels = sparse_nll_loss(
            probs=self.start_probs, labels=self.start_label)
        self.end_loss, _ = sparse_nll_loss(
            probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def get_loss(self):
        return self.loss
