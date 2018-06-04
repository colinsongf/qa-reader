"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-08 10:17:05
	* @modify date 2018-05-08 10:17:05
	* @desc [implement the R-NET algorithm described in https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf]
"""
import tensorflow as tf
from utils import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net


class Rnet(object):
    def __init__(self, vocab, config):
        self.is_train = True
        self.config = config
        self.vocab = vocab
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.hidden_size = config.hidden_size
        self.max_p_len = config.max_p_len
        self.max_q_len = config.max_q_len
        self.max_char_len = config.max_char_len

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            self.vocab.word_embeddings, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(self.vocab.char_embeddings, dtype=tf.float32))

        self.p = tf.placeholder(
            tf.int32, [None, None], "context")
        self.q = tf.placeholder(
            tf.int32, [None, None], "question")
        self.p_len = tf.placeholder(
            tf.int32, [None], "context_length")
        self.q_len = tf.placeholder(
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

        self.p_len = tf.reduce_sum(tf.cast(self.p_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.ph_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ph, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

    def ready(self):
        config = self.config
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.max_p_len, self.max_q_len, config.max_char_len, config.hidden_size, config.char_embed_dim, config.hidden_size
        # gru = cudnn_gru if config.use_cudnn else native_gru
        gru = native_gru

        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ph_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ph), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
                ph_emb = dropout(
                    ph_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                qh_emb = dropout(
                    qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ph_emb, self.ph_len, dtype=tf.float32)
                ph_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ph_emb = tf.reshape(ph_emb, [N, PL, 2 * dg])

            with tf.name_scope("word"):
                p_emb = tf.nn.embedding_lookup(self.word_mat, self.p)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            p_emb = tf.concat([p_emb, ph_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=p_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            p = rnn(p_emb, seq_len=self.p_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qp_att = dot_attention(p, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qp_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qp_att, seq_len=self.p_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.p_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.p_len)

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, self.p_mask)
            self.start_logits = logits1
            self.end_logits = logits2

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.start_probs = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.end_probs = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

            self.start_labels = tf.one_hot(
                self.start_label, tf.shape(logits1)[1], axis=1)
            self.end_labels = tf.one_hot(
                self.end_label, tf.shape(logits1)[1], axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=tf.stop_gradient(self.start_labels))
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=tf.stop_gradient(self.end_labels))
            self.loss = tf.reduce_mean(losses + losses2)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
