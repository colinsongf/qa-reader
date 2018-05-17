"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-08 02:34:43
	* @modify date 2018-05-09 07:01:49
	* @desc [This module implements cmrc dataset preprocess strategy]
"""

import jieba
import json
import re

with open('../data/extra.dict', 'r') as fin:
    for word in fin:
        word = word.strip()
        jieba.add_word(word)


def re_split(str, delimeters):
    delimeters = [re.escape(delimeter) for delimeter in delimeters]
    pattern = r'(' + '|'.join(delimeters) + ')'
    split_context_text = re.split(pattern, str)
    split_context_text = [part_context_text for part_context_text in split_context_text
                          if part_context_text is not None]
    split_context_text = [part_context_text for part_context_text in split_context_text
                          if part_context_text is not '']
    return split_context_text


def rebuild_delimeters(delimeters):
    delimeters.sort(key=lambda x: len(x))
    # origin_delimeters = set(delimeters)
    for i, d in enumerate(delimeters):
        for j in range(i + 1, len(delimeters)):
            if d in delimeters[j]:
                deli = '@@' + d + '@@'
                delimeters[j] = deli.join(delimeters[j].split(d))
    reb_delineters = set()
    for deli in delimeters:
        for d in deli.split('@@'):
            if d != '':
                reb_delineters.add(d)
    return list(reb_delineters)


def segment(sample):
    result = []
    context_text = sample['context_text']
    delimeters = [str(qa['answers'][0]) for qa in sample['qas']]
    # delimeters = rebuild_delimeters(delimeters)
    split_context_text = re_split(context_text, delimeters)
    context_text_tokens = []
    context_text_tokens_list = [list(jieba.cut(part_context_text))
                                for part_context_text in split_context_text]
    for l in context_text_tokens_list:
        context_text_tokens.extend(l)
    context_text_chars = [list(token) for token in context_text_tokens]
    title_tokens = list(jieba.cut(sample['title']))
    segmented_qas = []
    for qa in sample['qas']:
        ssample = {}
        ssample['context_id'] = sample['context_id']
        ssample['title'] = sample['title']
        ssample['context_text'] = sample['context_text']
        ssample['query_text'] = qa['query_text']
        ssample['query_id'] = qa['query_id']
        ssample['answers'] = qa['answers']

        query_text_tokens = list(jieba.cut(qa['query_text']))
        query_text_chars = [list(token) for token in query_text_tokens]
        answer_tokens = list(jieba.cut(str(qa['answers'][0])))
        ssample['query_text_tokens'] = query_text_tokens
        ssample['answer_tokens'] = answer_tokens
        ssample['context_text_tokens'] = context_text_tokens
        ssample['title_tokens'] = title_tokens
        ssample['context_text_chars'] = context_text_chars
        ssample['query_text_chars'] = query_text_chars
        result.append(ssample)
    return result


def segment_all():
    files = ['../data/raw/cmrc2018/train.json',
             '../data/raw/cmrc2018/dev.json',
             '../data/raw/cmrc2018/test.json']
    for file in files:
        new_file = file.replace('.json', '.segmented.json')
        with open(file, 'r') as fin:
            data = json.load(fin)
        result = []
        for sample in data:
            result.extend(segment(sample))
        with open(new_file, 'w') as fout:
            json.dump(result, fout, ensure_ascii=False)


def knuth_morris_pratt(text, pattern):
    """
    Yields all starting positions of copies of the pattern in the text.
    Calling conventions are similar to string.find, but its arguments can be
    lists or iterators, not just strings, it returns all matches, not just
    the first one, and it does not need the whole text in memory at once.
    Whenever it yields, it will have read the text exactly up to and including
    the match that caused the yield.
    """

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos - shift]:
            shift += shifts[pos - shift]
        shifts[pos + 1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
                matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


def longest_common_subsequence(a, b):
    """
    Longest common subsequence
    """
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j +
                               1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    index = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            index.append(x - 1)
            x -= 1
            y -= 1
    return result, index


def longest_common_substring(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return p - mmax, p


def find_answer_span(sample):
    context_text_tokens = sample['context_text_tokens']

    answer_span = []
    answer_tokens = sample['answer_tokens']
    query_text_tokens = sample['query_text_tokens']
    start_pos = knuth_morris_pratt(
        context_text_tokens, answer_tokens)
    start_pos = list(start_pos)
    if len(start_pos) == 0:
        start, end = longest_common_substring(
            context_text_tokens, answer_tokens)
        start = start - 2 if start - 2 >= 0 else start
        end = end + 2 if end + 2 < len(context_text_tokens) else end
        answer_span.append(start)
        answer_span.append(end)
    elif len(start_pos) == 1:
        start = start_pos[0]
        end = start + len(answer_tokens)
        answer_span.append(start)
        answer_span.append(end)
    else:
        best_start = -1
        max_match = 0
        for start in start_pos:
            end = start + len(answer_tokens)
            start = start - 5 if start - 5 >= 0 else start
            end = end + 5 if end + 5 < len(context_text_tokens) else end
            fake_query_text = context_text_tokens[start:end]
            _, index = longest_common_subsequence(
                query_text_tokens, fake_query_text)
            if len(index) > max_match:
                best_start = start
        start = best_start
        end = start + len(answer_tokens)
        answer_span.append(start)
        answer_span.append(end)

    sample['answer_span'] = answer_span
    return sample


def split_qas(data):
    result = []
    for sample in data:
        for qa in sample['segmented_qas']:
            new_sample = {}
            new_sample['context_id'] = sample['context_id']
            new_sample['segmented_title'] = sample['segmented_title']
            new_sample['segmented_context_text'] = sample['segmented_context_text']
            new_sample['segmented_query_text'] = qa['segmented_query_text']
            new_sample['segmented_answer'] = qa['segmented_answer']
            new_sample['query_id'] = qa['query_id']
            new_sample['answer_span'] = qa['answer_span']
            result.append(new_sample)
    return result


def find_answer_span_all():
    files = {'../data/raw/cmrc2018/train.segmented.json': '../data/preprocessed/cmrc2018/trainset/train.json',
             '../data/raw/cmrc2018/dev.segmented.json': '../data/preprocessed/cmrc2018/devset/dev.json',
             '../data/raw/cmrc2018/test.segmented.json': '../data/preprocessed/cmrc2018/testset/test.json'}
    for file, new_file in files.items():
        with open(file, 'r') as fin:
            data = json.load(fin)

        result = []
        for sample in data:
            result.append(find_answer_span(sample))

        with open(new_file, 'w') as fout:
            json.dump(result, fout, ensure_ascii=False)

        demo_file = new_file.replace('preprocessed', 'demo')
        with open(demo_file, 'w') as fout:
            json.dump(result[:100], fout, ensure_ascii=False)


def main():
    segment_all()
    find_answer_span_all()


if __name__ == '__main__':
    main()
