"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-15 03:26:55
	* @modify date 2018-05-16 09:33:33
	* @desc [description]
"""

import os
import json
import logging
from general import get_feed_dict
from utils import compute_bleu_rouge
from utils import normalize
from utils import evaluate_


class Evaluator(object):
    def __init__(self, sess, model, config):
        self.sess = sess
        self.model = model
        self.config = config
        self.start_probs = model.start_probs
        self.end_probs = model.end_probs
        self.loss = model.loss
        self.max_p_len = config.max_p_len
        self.max_a_len = config.max_a_len
        self.logger = logging.getLogger('qarc')

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
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            # print('start_probs:', start_probs.shape)
            # print('end_probs:', end_probs.shape)

            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):
                best_answer, max_prob = self.find_best_answer(
                    sample, start_prob, end_prob)
                answers = [''.join(sample['segmented_answer'])]
                ref_answers.append({'query_id': sample['query_id'],
                                    'answers': answers})
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'query_id': sample['query_id'],
                                         'answers': [best_answer],
                                         'ground_truth': answers})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(
                        pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(
                result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                query_id = ref['query_id']
                if len(ref['answers']) > 0:
                    pred_dict[query_id] = pred['answers']
                    ref_dict[query_id] = ref['answers']
            F1, EM, TOTAL, SKIP = evaluate_(ref_dict, pred_dict)
            AVG = (EM + F1) * 0.5
        else:
            F1, EM, TOTAL, SKIP = None, None, None, None
        return ave_loss, F1, EM, AVG

    def find_best_answer(self, sample, start_prob, end_prob):

        passage_len = min(self.max_p_len, len(
            sample['context_text_tokens']), len(start_prob))
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_prob[start_idx] * end_prob[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        best_answer = ''.join(
            sample['context_text_tokens'][best_start:best_end + 1])
        return best_answer, max_prob
