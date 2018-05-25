"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-25 01:08:45
	* @modify date 2018-05-25 01:08:45
	* @desc [implement a base rc model]
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf


class BaseModel(object):
    def __init__(self, vocab, config):

        # logging
        self.logger = logging.getLogger('qarc')

        # basic config
        self.hidden_size = config.hidden_size

        # length limit
        self.max_p_len = config.max_p_len
        self.max_q_len = config.max_q_len
        self.max_a_len = config.max_a_len

        # the vocab
        self.vocab = vocab
        self.word_embed_dim = vocab.word_embed_dim

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
        # self._predict()
        self._create_summaries()
        self.logger.info(
            'Time to build graph: {} s'.format(time.time() - start_t))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None], name="passage")
        self.q = tf.placeholder(tf.int32, [None, None], name="question")
        self.p_len = tf.placeholder(tf.int32, [None], name="passage_len")
        self.q_len = tf.placeholder(tf.int32, [None], name="question_len")
        self.start_label = tf.placeholder(tf.int32, [None], name="start_label")
        self.end_label = tf.placeholder(tf.int32, [None], name="end_label")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.dropout_keep_prob = 1.0 - 0.5 * self.dropout

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """

        with tf.variable_scope('embeddings'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.word_size(), self.vocab.word_embed_dim),
                initializer=tf.constant_initializer(
                    self.vocab.word_embeddings),
                trainable=False
            )
            self.p_emb = tf.nn.embedding_lookup(
                self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(
                self.word_embeddings, self.q)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            p_cell_fw = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            p_cell_bw = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            p_output, _ = tf.nn.bidirectional_dynamic_rnn(
                p_cell_fw, p_cell_bw, self.p_emb, sequence_length=self.p_len, dtype=tf.float32)
            self.p_encodes = tf.concat(p_output, 2)
        with tf.variable_scope('question_encoding'):
            q_cell_fw = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            q_cell_bw = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            q_output, _ = tf.nn.bidirectional_dynamic_rnn(
                q_cell_fw, q_cell_bw, self.q_emb, sequence_length=self.q_len, dtype=tf.float32)
            self.q_encodes = tf.concat(q_output, 2)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """

        p_mask = tf.sequence_mask(self.p_len, tf.shape(
            self.p)[1], dtype=tf.float32, name='passage_mask')
        q_mask = tf.sequence_mask(self.q_len, tf.shape(
            self.q)[1], dtype=tf.float32, name='question_mask')
        sim_matrix = tf.matmul(
            self.p_encodes, self.q_encodes, transpose_b=True)
        sim_mask = tf.matmul(tf.expand_dims(p_mask, -1),
                             tf.expand_dims(q_mask, -1), transpose_b=True)
        # mask out zeros by replacing it with very small number
        sim_matrix -= (1 - sim_mask) * 1e30
        passage2question_attn = tf.matmul(
            tf.nn.softmax(sim_matrix, -1), self.q_encodes)
        b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
        question2passage_attn = tf.tile(tf.matmul(b, self.p_encodes),
                                        [1, tf.shape(self.p_encodes)[1], 1])
        self.p_mask = tf.expand_dims(p_mask, -1)
        passage2question_attn *= self.p_mask
        question2passage_attn *= self.p_mask
        self.match_out = tf.concat([self.p_encodes,
                                    self.p_encodes * passage2question_attn,
                                    self.p_encodes * question2passage_attn], -1)

    def _fuse(self):
        out_dim = 64
        window_len = 10
        conv_match = tf.layers.conv1d(
            self.match_out, out_dim, window_len, strides=window_len)
        conv_match_up = tf.squeeze(tf.image.resize_images(tf.expand_dims(conv_match, axis=-1),
                                                          [tf.shape(self.match_out)[
                                                              1], out_dim],
                                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), axis=-1)
        self.fuse_out = tf.concat(
            [self.p_encodes, self.match_out, conv_match_up], axis=-1)

    def _decode(self):
        self.start_logit = tf.layers.dense(self.fuse_out, 1)
        self.end_logit = tf.layers.dense(self.fuse_out, 1)
        # mask out those padded symbols before softmax
        self.start_logit -= (1 - self.p_mask) * 1e30
        self.end_logit -= (1 - self.p_mask) * 1e30
        self.start_probs = tf.squeeze(tf.nn.softmax(self.start_logit, dim=1))
        self.end_probs = tf.squeeze(tf.nn.softmax(self.end_logit, dim=1))

    def _compute_loss(self):
        # compute the loss
        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses
        self.start_loss = sparse_nll_loss(
            probs=self.start_probs, labels=self.start_label)
        self.start_loss = tf.reduce_mean(self.start_loss)
        self.end_loss = sparse_nll_loss(
            probs=self.end_probs, labels=self.end_label)
        self.end_loss = tf.reduce_mean(self.end_loss)
        self.loss = tf.add(self.start_loss, self.end_loss)

        # self.start_loss = tf.losses.sparse_softmax_cross_entropy(
        #     labels=self.start_label, logits=self.start_logit)
        # self.end_loss = tf.losses.sparse_softmax_cross_entropy(
        #     labels=self.end_label, logits=self.end_logit)
        # self.loss = (self.start_loss + self.end_loss) / 2

    def _predict(self):

        # do the outer product
        outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
                          tf.expand_dims(end_prob, axis=1))
        outer = tf.matrix_band_part(outer, 0, self.max_a_len)
        self.start_pos = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.end_pos = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('start_loss', tf.squeeze(self.start_loss))
            tf.summary.scalar('end_loss', tf.squeeze(self.end_loss))
            tf.summary.scalar('loss', self.loss)
            # tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def get_loss(self):
        return self.loss
