"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-15 03:26:55
	* @modify date 2018-05-15 03:26:55
	* @desc [description]
"""

from general import get_feed_dict


class Evaluator(object):
    def __init__(self, sess, model, config):
        self.sess = sess
        self.model = model
        self.config = config
        self.start_probs = model.start_probs
        self.end_probs = model.end_probs
        self.loss = model.loss

        self.p_emb = model.p_emb
        self.q_emb = model.q_emb
        self.sep_q_encodes = model.sep_q_encodes
        self.sep_p_encodes = model.sep_p_encodes
        self.match_p_encodes = model.match_p_encodes
        self.fuse_p_encodes = model.fuse_p_encodes
        self.concat_passage_encodes = model.concat_passage_encodes
        self.no_dup_question_encodes = model.no_dup_question_encodes

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = get_feed_dict(
                self.model, batch, self.config.dropout_keep_prob)
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            p_emb, q_emb, \
                sep_q_encodes, \
                sep_p_encodes, \
                match_p_encodes, \
                fuse_p_encodes, \
                concat_passage_encodes, \
                no_dup_question_encodes = self.sess.run([self.p_emb,
                                                         self.q_emb,
                                                         self.sep_q_encodes,
                                                         self.sep_p_encodes,
                                                         self.match_p_encodes,
                                                         self.fuse_p_encodes,
                                                         self.concat_passage_encodes,
                                                         self.no_dup_question_encodes], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            print('start_probs:', start_probs.shape)
            print('end_probs:', end_probs.shape)

            print('p_emb:', p_emb.shape)
            print('q_emb:', q_emb.shape)
            print('sep_q_encodes:', sep_q_encodes.shape)
            print('sep_p_encodes:', sep_p_encodes.shape)
            print('match_p_encodes:', match_p_encodes.shape)
            print('fuse_p_encodes:', fuse_p_encodes.shape)
            print('concat_passage_encodes:', concat_passage_encodes.shape)
            print('no_dup_question_encodes:', no_dup_question_encodes.shape)
            print('********************************')
