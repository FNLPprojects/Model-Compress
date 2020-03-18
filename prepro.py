# -*- coding:utf-8 -*-
'''
@Author:LinXinzhu
@email:18210980055@fudan.edu.cn
'''

import tensorflow as tf
import os
import json
'''
创建词表，输入词表大小，输出word2id和id2word字典，以及路径
'''
def create_vocabulary(vocab_size, ids_dir='./ids/'):

    if not tf.gfile.Exists(ids_dir):
        tf.gfile.MakeDirs(ids_dir)

    vocab_ids_path = os.path.join(ids_dir, ('vocabs%d.ids' % vocab_size))

    if not tf.gfile.Exists(vocab_ids_path):
        print('Vocab file %s not found. Creating new vocabs.ids file' % vocab_ids_path)

        word2freq = {}

        with open('./data/sst2/sentiment.train.0') as fin:
            for line in fin.readlines():
                for word in line.strip().split():
                    word2freq[word] = word2freq.get(word, 0) + 1

        with open('./data/sst2/sentiment.train.1') as fin:
            for line in fin.readlines():
                for word in line.strip().split():
                    word2freq[word] = word2freq.get(word, 0) + 1

        sorted_dict = sorted(word2freq.items(), key=lambda item: item[1], reverse=True)
        sorted_dict = sorted_dict[:vocab_size - 4]

        word2idx = {'_PAD_': 0, '_UNK_': 1, '_BOS_': 2, '_EOS_': 3,}
        idx2word = ['_PAD_', '_UNK_', '_BOS_', '_EOS_']
        for w, _ in sorted_dict:
            if w not in ['_PAD_', '_UNK_', '_BOS_', '_EOS_']:
                word2idx[w] = len(word2idx)
                idx2word.append(w)

        print('Save vocabularies into file %s ...' % vocab_ids_path)
        json.dump({'word2idx': word2idx, 'idx2word': idx2word}, open(vocab_ids_path, 'w'), ensure_ascii=False)

        return word2idx, idx2word, vocab_ids_path

    else:
        print('Loading Vocabularies from %s ...' % vocab_ids_path)
        d_vocab = json.load(open(vocab_ids_path))
        word2idx = d_vocab['word2idx']
        idx2word = d_vocab['idx2word']

        return word2idx, idx2word, vocab_ids_path
'''
创建蒸馏词表，输入词表大小，输出word2id和id2word字典，以及路径
'''
def create_vocabulary_distil(vocab_size, ids_dir='./ids/'):

    if not tf.gfile.Exists(ids_dir):
        tf.gfile.MakeDirs(ids_dir)

    vocab_ids_path = os.path.join(ids_dir, ('vocabs%d.ids' % vocab_size))

    if not tf.gfile.Exists(vocab_ids_path):
        print('Vocab file %s not found. Creating new vocabs.ids file' % vocab_ids_path)

        word2freq = {}

        with open('./data/sst2/augmented.tsv') as fin:
            count=0
            for line in fin.readlines():
                count += 1
                if count == 1:
                    continue
                wordlist=line.strip().split()
                wordlist=wordlist[:len(wordlist)-2]
                for word in wordlist:
                    word2freq[word] = word2freq.get(word, 0) + 1


        sorted_dict = sorted(word2freq.items(), key=lambda item: item[1], reverse=True)
        sorted_dict = sorted_dict[:vocab_size - 4]

        word2idx = {'_PAD_': 0, '_UNK_': 1, '_BOS_': 2, '_EOS_': 3,}
        idx2word = ['_PAD_', '_UNK_', '_BOS_', '_EOS_']
        for w, _ in sorted_dict:
            if w not in ['_PAD_', '_UNK_', '_BOS_', '_EOS_']:
                word2idx[w] = len(word2idx)
                idx2word.append(w)

        print('Save vocabularies into file %s ...' % vocab_ids_path)
        json.dump({'word2idx': word2idx, 'idx2word': idx2word}, open(vocab_ids_path, 'w'), ensure_ascii=False)

        return word2idx, idx2word, vocab_ids_path

    else:
        print('Loading Vocabularies from %s ...' % vocab_ids_path)
        d_vocab = json.load(open(vocab_ids_path))
        word2idx = d_vocab['word2idx']
        idx2word = d_vocab['idx2word']

        return word2idx, idx2word, vocab_ids_path
#格式转换
def createdata():
    if not tf.gfile.Exists('./data/sst2/train.tsv'):
        print("no SST2 data")
    else:
        data0 = []
        data1 = []
        count=0
        with open('./data/sst2/train.tsv') as fin:
            for line in fin.readlines():
                count+=1
                if count==1:
                    continue
                words = line.strip().split()
                w=words[:len(words)-1]
                label=words[-1]
                if label=='0':
                    data0.append(" ".join(w))
                if label=='1':
                    data1.append(" ".join(w))
                if label!='0' and label!='1':
                    print("error")
                    raise ConnectionAbortedError
        with open('./data/sst2/sentiment.train.0', 'w') as fout:
            for d in data0:
                fout.write(d)
                fout.write('\n')
        with open('./data/sst2/sentiment.train.1', 'w') as fout:
            for d in data1:
                fout.write(d)
                fout.write('\n')
        #dev
        data0 = []
        data1 = []
        count = 0
        with open('./data/sst2/dev.tsv') as fin:
            for line in fin.readlines():
                count += 1
                if count == 1:
                    continue
                words = line.strip().split()
                w = words[:len(words) - 1]
                label = words[-1]
                if label == '0':
                    data0.append(" ".join(w))
                if label == '1':
                    data1.append(" ".join(w))
                if label != '0' and label != '1':
                    print("error")
                    raise ConnectionAbortedError
        with open('./data/sst2/sentiment.dev.0', 'w') as fout:
            for d in data0:
                fout.write(d)
                fout.write('\n')
        with open('./data/sst2/sentiment.dev.1', 'w') as fout:
            for d in data1:
                fout.write(d)
                fout.write('\n')



'''
创建数据集的索引文件
'''
def create_sst2_ids(word2idx):

    if not tf.gfile.Exists('./ids/sst2/'):
        tf.gfile.MakeDirs('./ids/sst2/')

    if not tf.gfile.Exists('./ids/sst2/sentiment.train.0.ids'):
        print('Ids file for sst2/train.0 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.train.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.train.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.train.1.ids'):
        print('Ids file for sst2/train.1 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.train.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.train.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.dev.0.ids'):
        print('Ids file for sst2/dev.0 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.dev.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.dev.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.dev.1.ids'):
        print('Ids file for sst2/dev.1 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.dev.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.dev.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.test.0.ids'):
        print('Ids file for sst2/test.0 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.test.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.test.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.test.1.ids'):
        print('Ids file for sst2/test.1 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.test.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.test.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

'''
创建词表索引
'''
def create_sst2_ids_distil(word2idx):

    if not tf.gfile.Exists('./ids/sst2/'):
        tf.gfile.MakeDirs('./ids/sst2/')

    if not tf.gfile.Exists('./ids/sst2/augmented.ids'):
        print('Ids file for augmented.ids not found. Creating ...')

        data = []
        with open('./data/sst2/augmented.tsv') as fin:
            count=0
            for line in fin.readlines():
                count += 1
                if count == 1:
                    continue
                words = line.strip().split()
                words = words[:len(words)-2]
                if len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/augmented.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')


    if not tf.gfile.Exists('./ids/sst2/sentiment.dev.0.ids'):
        print('Ids file for sst2/dev.0 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.dev.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.dev.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.dev.1.ids'):
        print('Ids file for sst2/dev.1 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.dev.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.dev.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.test.0.ids'):
        print('Ids file for sst2/test.0 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.test.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.test.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/sst2/sentiment.test.1.ids'):
        print('Ids file for sst2/test.1 not found. Creating ...')

        data = []
        with open('./data/sst2/sentiment.test.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/sst2/sentiment.test.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')