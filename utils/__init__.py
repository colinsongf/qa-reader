# coding:utf8
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This package implements some utility functions shared by PaddlePaddle
and Tensorflow model implementations.

Authors: liuyuan(liuyuan04@baidu.com)
Date:    2017/10/06 18:23:06
"""


from .dureader_eval import compute_bleu_rouge
from .dureader_eval import normalize
from .cmrc2018_evaluate import evaluate_
from .cmrc2018_evaluate import evaluate
from .preprocess import find_fake_answer
from .preprocess import find_best_question_match
from .vec_helper import BenebotVector
from .parameter import Config
from .interface_utils import find_best_answer
from .interface_utils import get_feature
from .network import rnn
from .network import MatchLSTMLayer
from .network import AttentionFlowMatchLayer
from .network import PointerNetDecoder
from .network import highway, conv, initializer, residual_block, regularizer
from .network import mask_logits, trilinear, total_params, optimized_trilinear_for_attention
from .network import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net
from .vocab import Vocab
from .brc_dataset import BRCDataset
from .cmrc_dataset import CMRCDataset
from .search_utils import SolrSearch, BaiduSearch

__all__ = [
    'compute_bleu_rouge',
    'evaluate_',
    'evaluate',
    'normalize',
    'find_fake_answer',
    'find_best_question_match',
    'BenebotVector',
    'Config',
    'find_best_answer',
    'get_feature',
    'rnn',
    'MatchLSTMLayer',
    'AttentionFlowMatchLayer',
    'PointerNetDecoder',
    'highway',
    'conv',
    'initializer',
    'residual_block',
    'mask_logits',
    'trilinear',
    'total_params',
    'optimized_trilinear_for_attention',
    'regularizer',
    'Vocab',
    'BRCDataset',
    'CMRCDataset',
    'SolrSearch',
    'BaiduSearch',
    'cudnn_gru',
    'native_gru',
    'dot_attention',
    'summ',
    'dropout',
    'ptr_net'
]
