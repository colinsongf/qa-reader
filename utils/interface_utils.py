"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-24 10:00:53
	* @modify date 2018-05-24 10:00:53
	* @desc [some utils used in interface]
"""

import jieba


def get_feed_dict(model, batch, dropout=0.1):
    feed_dict = {model.p: batch['passage_token_ids'],
                 model.q: batch['question_token_ids'],
                 model.p_len: batch['passage_length'],
                 model.q_len: batch['question_length'],
                 #  model.ph: batch['passage_char_ids'],
                 #  model.qh: batch['question_char_ids'],
                 model.ppy: batch['passage_py_ids'],
                 model.qpy: batch['question_py_ids'],
                 model.start_label: batch['start_id'],
                 model.end_label: batch['end_id'],
                 model.dropout: dropout}


def find_best_answer(start_prob, end_prob, max_a_len):

    passage_len = len(start_prob)
    best_start, best_end, max_prob = -1, -1, 0
    for start_idx in range(passage_len):
        for ans_len in range(max_a_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_prob[start_idx] * end_prob[end_idx]
            if prob > max_prob:
                best_start = start_idx
                best_end = end_idx
                max_prob = prob
    return best_start, best_end, max_prob


def get_feature(model, inputs, vocab, config):
    max_p_len = config.max_p_len
    max_q_len = config.max_q_len
    pad_char_len = config.max_char_len
    pad_token_id = vocab.get_word_id(vocab.pad_token)
    pad_char_id = vocab.get_char_id(vocab.pad_char)

    passage_tokens = list(jieba.cut(inputs[0]))
    question_tokens = list(jieba.cut(inputs[1]))
    p_len = len(passage_tokens)
    q_len = len(question_tokens)
    passage_token_ids = vocab.convert_word_to_ids(
        passage_tokens)
    passage_token_ids = passage_token_ids + \
        ([pad_token_id] * (max_p_len - len(passage_token_ids)
                           ))[: max_p_len]
    question_token_ids = vocab.convert_word_to_ids(
        question_tokens)
    question_token_ids = question_token_ids + \
        ([pad_token_id] * (max_q_len - len(question_token_ids)
                           ))[: max_q_len]

    passage_char_ids = [vocab.convert_char_to_ids(
        list(word)) for word in passage_tokens]
    passage_char_ids = passage_char_ids + \
        ([[pad_char_id]] * (max_p_len - len(passage_char_ids)
                            ))[: max_p_len]
    passage_char_ids = [(ids + [pad_char_id] * (pad_char_len - len(ids)))[
        : pad_char_len] for ids in passage_char_ids]

    question_char_ids = [vocab.convert_char_to_ids(
        list(word)) for word in question_tokens]
    question_char_ids = question_char_ids + \
        ([[pad_char_id]] * (max_q_len - len(question_char_ids)
                            ))[: max_q_len]
    question_char_ids = [(ids + [pad_char_id] * (pad_char_len - len(ids)))[
        : pad_char_len] for ids in question_char_ids]

    feed_dict = {model.p: [passage_token_ids],
                 model.q: [question_token_ids],
                 model.p_len: [p_len],
                 model.q_len: [q_len],
                 model.ph: [passage_char_ids],
                 model.qh: [question_char_ids],
                 model.start_label: [0],
                 model.end_label: [0],
                 model.dropout: config.dropout}

    return feed_dict
