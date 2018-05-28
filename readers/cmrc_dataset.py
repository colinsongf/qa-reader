"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-10 08:52:22
	* @modify date 2018-05-10 08:52:22
	* @desc [This module implements cmrc data process strategies.]
"""

import os
import json
import logging
import numpy as np
from collections import Counter
from tqdm import tqdm


class CMRCDataset(object):

    def __init__(self, max_p_len, max_q_len, max_char_len, max_py_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger('qarc')
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_char_len = max_char_len
        self.max_py_len = max_py_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            self.logger.info('Load train dataset...')
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info(
                'Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            self.logger.info('Load dev dataset...')
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info(
                'Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            self.logger.info('Load test dataset...')
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info(
                'Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        """
        # data_set = []
        with open(data_path, 'r') as fin:
            data_set = json.load(fin)
        # for sample in data:
        #     del sample['context_text']
        #     del sample['qas']
        #     del sample['title']
        #     data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """

        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_char_ids': [],
                      'question_py_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_char_ids': [],
                      'passage_py_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        for sidx, sample in enumerate(batch_data['raw_data']):
            question_token_ids = sample['question_token_ids']
            question_char_ids = sample['question_char_ids']
            question_py_ids = sample['question_py_ids']
            passage_token_ids = sample['passage_token_ids']
            passage_char_ids = sample['passage_char_ids']
            passage_py_ids = sample['passage_py_ids']
            answer_span = sample['answer_span']
            batch_data['passage_token_ids'].append(passage_token_ids)
            batch_data['passage_char_ids'].append(passage_char_ids)
            batch_data['passage_py_ids'].append(passage_py_ids)
            batch_data['question_token_ids'].append(question_token_ids)
            batch_data['question_char_ids'].append(question_char_ids)
            batch_data['question_py_ids'].append(question_py_ids)
            batch_data['question_length'].append(len(question_token_ids))
            batch_data['passage_length'].append(
                min(len(passage_token_ids), self.max_p_len))
            batch_data['start_id'].append(answer_span[0])
            batch_data['end_id'].append(answer_span[1])

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(
            batch_data, pad_id)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        # pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        # pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (self.max_p_len - len(ids)))[: self.max_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (self.max_q_len - len(ids)))[: self.max_q_len]
                                            for ids in batch_data['question_token_ids']]
        pad_char_len = self.max_char_len
        pad_py_len = self.max_py_len

        batch_data['question_char_ids'] = [(id_list + [[pad_id]] * (self.max_q_len - len(id_list)))[
            :self.max_q_len] for id_list in batch_data['question_char_ids']]
        question_char_ids_list = []
        for id_list in batch_data['question_char_ids']:
            question_char_ids_list.append([(ids + [pad_id] * (pad_char_len - len(ids)))[
                :pad_char_len] for ids in id_list])
        batch_data['question_char_ids'] = question_char_ids_list

        batch_data['question_py_ids'] = [(id_list + [[pad_id]] * (self.max_q_len - len(id_list)))[
            :self.max_q_len] for id_list in batch_data['question_py_ids']]
        question_py_ids_list = []
        for id_list in batch_data['question_py_ids']:
            question_py_ids_list.append([(ids + [pad_id] * (pad_py_len - len(ids)))[
                :pad_py_len] for ids in id_list])
        batch_data['question_py_ids'] = question_py_ids_list

        batch_data['passage_char_ids'] = [(id_list + [[pad_id]] * (self.max_p_len - len(id_list)))[
            :self.max_p_len] for id_list in batch_data['passage_char_ids']]
        passage_char_ids_list = []
        for id_list in batch_data['passage_char_ids']:
            passage_char_ids_list.append([(ids + [pad_id] * (pad_char_len - len(ids)))[
                :pad_char_len] for ids in id_list])
        batch_data['passage_char_ids'] = passage_char_ids_list

        batch_data['passage_py_ids'] = [(id_list + [[pad_id]] * (self.max_p_len - len(id_list)))[
            :self.max_p_len] for id_list in batch_data['passage_py_ids']]
        passage_py_ids_list = []
        for id_list in batch_data['passage_py_ids']:
            passage_py_ids_list.append([(ids + [pad_id] * (pad_py_len - len(ids)))[
                :pad_py_len] for ids in id_list])
        batch_data['passage_py_ids'] = passage_py_ids_list

        # print(np.asanyarray(passage_char_ids_list).shape)

        return batch_data, self.max_p_len, self.max_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                # print(sample)
                for token in sample['context_text_tokens']:
                    yield token
                for token in sample['query_text_tokens']:
                    yield token

    def char_iter(self, set_name=None):
        """
        Iterates over all the char in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                # print(sample)
                for token in sample['context_text_chars']:
                    for char in token:
                        yield char
                for token in sample['query_text_chars']:
                    for char in token:
                        yield char

    def py_iter(self, set_name=None):
        """
        Iterates over all the char in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                # print(sample)
                for token in sample['context_text_pys']:
                    for py in token:
                        yield py
                for token in sample['query_text_pys']:
                    for py in token:
                        yield py

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['passage_token_ids'] = vocab.convert_word_to_ids(
                    sample['context_text_tokens'])
                sample['question_token_ids'] = vocab.convert_word_to_ids(
                    sample['query_text_tokens'])

                sample['passage_char_ids'] = []
                sample['question_char_ids'] = []
                for chars in sample['context_text_chars']:
                    sample['passage_char_ids'].append(
                        vocab.convert_char_to_ids(chars))
                for chars in sample['query_text_chars']:
                    sample['question_char_ids'].append(
                        vocab.convert_char_to_ids(chars))

                sample['passage_py_ids'] = []
                sample['question_py_ids'] = []
                for pys in sample['context_text_pys']:
                    sample['passage_py_ids'].append(
                        vocab.convert_py_to_ids(pys))
                for pys in sample['query_text_pys']:
                    sample['question_py_ids'].append(
                        vocab.convert_py_to_ids(pys))

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
