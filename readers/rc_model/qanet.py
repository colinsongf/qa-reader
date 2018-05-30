"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-08 10:13:04
	* @modify date 2018-05-08 10:13:04
	* @desc [implement the QANet algorithm described in https://openreview.net/pdf?id=B14TlG-RW]
"""

"""
https://github.com/NLPLearn/QANet
"""

import logging
import time
import tensorflow as tf
from utils import initializer, regularizer, residual_block, highway, conv
from utils import mask_logits, trilinear, total_params, optimized_trilinear_for_attention


class QANet(object):
    def __init__(self, vocab, config):

        # logging
        self.logger = logging.getLogger('qarc')

        self.config = config

        # basic config
        self.hidden_size = config.hidden_size
        self.num_head = config.num_head

        # length limit
        self.max_p_len = config.max_p_len
        self.max_q_len = config.max_q_len
        self.max_char_len = config.max_char_len

        self.vocab = vocab
        self.word_embed_dim = vocab.word_embed_dim
        self.char_embed_dim = vocab.char_embed_dim

        self._build_graph()

        self.global_step = tf.get_variable('global_step', shape=[
        ], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

    def _build_graph(self):
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_summaries()
        self.logger.info(
            'Time to build graph: {} s'.format(time.time() - start_t))

    def _setup_placeholders(self):
        self.p = tf.placeholder(
            tf.int32, [None, None], "context")
        self.q = tf.placeholder(
            tf.int32, [None, None], "question")
        self.p_length = tf.placeholder(
            tf.int32, [None], "context_length")
        self.q_length = tf.placeholder(
            tf.int32, [None], "question_length")
        self.ph = tf.placeholder(
            tf.int32, [None, None, None], "context_char")
        self.qh = tf.placeholder(
            tf.int32, [None, None, None], "question_char")
        self.start_label = tf.placeholder(
            tf.int32, [None], "answer_index1")
        self.end_label = tf.placeholder(
            tf.int32, [None], "answer_index2")
        self.dropout = tf.placeholder(tf.float32)

        self.p_mask = tf.cast(self.p, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)

    def _embed(self):
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.word_size(), self.vocab.word_embed_dim),
                initializer=tf.constant_initializer(
                    self.vocab.word_embeddings),
                trainable=False
            )

            self.char_embeddings = tf.get_variable(
                'char_embeddings',
                shape=(self.vocab.char_size(), self.vocab.char_embed_dim),
                initializer=tf.constant_initializer(
                    self.vocab.char_embeddings)
            )
        ph_emb = tf.reshape(tf.nn.embedding_lookup(
            self.char_embeddings, self.ph), [-1, self.max_char_len, self.char_embed_dim])
        qh_emb = tf.reshape(tf.nn.embedding_lookup(
            self.char_embeddings, self.qh), [-1, self.max_char_len, self.char_embed_dim])
        ph_emb = tf.nn.dropout(ph_emb, 1.0 - 0.5 * self.dropout)
        qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

        # Bidaf style conv-highway encoder
        ph_emb = conv(ph_emb, self.hidden_size,
                      bias=True, activation=tf.nn.relu, kernel_size=3, name="char_conv", reuse=None)
        qh_emb = conv(qh_emb, self.hidden_size,
                      bias=True, activation=tf.nn.relu, kernel_size=3, name="char_conv", reuse=True)

        ph_emb = tf.reduce_max(ph_emb, axis=1)
        qh_emb = tf.reduce_max(qh_emb, axis=1)

        ph_emb = tf.reshape(ph_emb, [-1, self.max_p_len, ph_emb.shape[-1]])
        qh_emb = tf.reshape(
            qh_emb, [-1, self.max_q_len, qh_emb.shape[-1]])

        p_emb = tf.nn.dropout(tf.nn.embedding_lookup(
            self.word_embeddings, self.p), 1.0 - 0.5 * self.dropout)
        q_emb = tf.nn.dropout(tf.nn.embedding_lookup(
            self.word_embeddings, self.q), 1.0 - 0.5 * self.dropout)

        p_emb = tf.concat([p_emb, ph_emb], axis=2)
        q_emb = tf.concat([q_emb, qh_emb], axis=2)

        self.p_emb = highway(p_emb, size=self.hidden_size, scope="highway",
                             dropout=self.dropout, reuse=None)
        self.q_emb = highway(q_emb, size=self.hidden_size, scope="highway",
                             dropout=self.dropout, reuse=True)

    def _encode(self):
        with tf.variable_scope("passage_question_encoding"):
            self.p_encode = residual_block(self.p_emb,
                                           num_blocks=1,
                                           num_conv_layers=4,
                                           kernel_size=7,
                                           mask=None,
                                           num_filters=self.hidden_size,
                                           num_heads=self.num_head,
                                           seq_len=self.p_length,
                                           scope="Encoder_Residual_Block",
                                           bias=False,
                                           dropout=self.dropout)
            self.q_encode = residual_block(self.q_emb,
                                           num_blocks=1,
                                           num_conv_layers=4,
                                           kernel_size=7,
                                           mask=None,
                                           num_filters=self.hidden_size,
                                           num_heads=self.num_head,
                                           seq_len=self.q_length,
                                           scope="Encoder_Residual_Block",
                                           reuse=True,  # Share the weights between passage and question
                                           bias=False,
                                           dropout=self.dropout)

    def _match(self):
        with tf.variable_scope("context_to_query_attention_layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention(
                [self.p_encode, self.q_encode], self.max_p_len, self.max_q_len, input_keep_prob=1.0 - self.dropout)
            # mask_q = tf.expand_dims(self.q_mask, 1)
            # S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            S_ = tf.nn.softmax(S)
            # mask_p = tf.expand_dims(self.p_mask, 2)
            # S_T = tf.transpose(tf.nn.softmax(
            # mask_logits(S, mask=mask_p), dim=1), (0, 2, 1))
            S_T = tf.transpose(tf.nn.softmax(S, dim=1), (0, 2, 1))
            self.p2q = tf.matmul(S_, self.q_encode)
            self.q2p = tf.matmul(tf.matmul(S_, S_T), self.p_encode)
            self.attention_outputs = [self.p_encode, self.p2q,
                                      self.p_encode * self.p2q, self.p_encode * self.q2p]

    def _fuse(self):
        with tf.variable_scope("model_encoder_layer"):
            inputs = tf.concat(self.attention_outputs, axis=-1)
            self.enc = [conv(inputs, self.hidden_size,
                             name="input_projection")]
            for i in range(3):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(
                        self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                                   num_blocks=7,
                                   num_conv_layers=2,
                                   kernel_size=5,
                                   mask=None,
                                   num_filters=self.hidden_size,
                                   num_heads=self.num_head,
                                   seq_len=self.p_length,
                                   scope="Model_Encoder",
                                   bias=False,
                                   reuse=True if i > 0 else None,
                                   dropout=self.dropout)
                )

    def _decode(self):
        with tf.variable_scope("output_layer"):
            start_logits = tf.squeeze(conv(tf.concat(
                [self.enc[1], self.enc[2]], axis=-1), 1, bias=False, name="start_pointer"), -1)
            end_logits = tf.squeeze(conv(tf.concat(
                [self.enc[1], self.enc[3]], axis=-1), 1, bias=False, name="end_pointer"), -1)
            # self.logits = [mask_logits(start_logits, mask=self.p_mask),
            #                mask_logits(end_logits, mask=self.p_mask)]

            self.start_probs, self.end_probs = start_logits, end_logits

    def _compute_loss(self):

        def loss(logits, labels):
            labels = tf.one_hot(labels, tf.shape(logits)[1], axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            return loss

        self.start_loss = tf.reduce_mean(
            loss(self.start_probs, self.start_label))
        self.end_loss = tf.reduce_mean(loss(self.end_probs, self.end_label))
        self.loss = self.start_loss + self.end_loss

        if self.config.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(
                regularizer, variables)
            self.loss += l2_loss

        if self.config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var, v))

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('start_loss', tf.squeeze(self.start_loss))
            tf.summary.scalar('end_loss', tf.squeeze(self.end_loss))
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
