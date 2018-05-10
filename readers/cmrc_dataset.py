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

    def __init__(self, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[])
        self.logger = logging.getLogger('qarc')
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

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
        data_set = []
        with open(data_path, 'r') as fin:
            data = json.load(fin)
        for sample in data:
            del sample['context_text']
            del sample['qas']
            del sample['title']
            data_set.append(sample)
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
        raw_data = [data[i] for i in indices]
        batch_data = {'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        for sidx, sample in enumerate(raw_data):
            passage_token_ids = sample['passage_token_ids']
            passage_length = sample['passage_length']
            for qaidx in range(len(sample['qas'])):
                question_token_ids = sample['segmented_qas'][qaidx]['question_token_ids']
                batch_data['question_token_ids'].append(question_token_ids)
                batch_data['question_length'].append(len(question_token_ids))
                batch_data['passage_token_ids'].append(passage_token_ids)
                batch_data['passage_length'].append(passage_length)
                answer_span = sample['segmented_qas'][qaidx]['answer_span']
                batch_data['start_id'].append(answer_span[0])
                batch_data['end_id'].append(answer_span[1])

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(
            batch_data, pad_id)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

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
                for token in sample['segmented_context_text']:
                    yield token
                for qa in sample['segmented_qas']:
                    for token in qa['segmented_query_text']:
                        yield token

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
                sample['passage_token_ids'] = vocab.convert_to_ids(
                    sample['segmented_context_text'])
                for qa in sample['segmented_qas']:
                    qa['question_token_ids'] = vocab.convert_to_ids(
                        qa['segmented_query_text'])

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
