# -*- coding:utf-8 -*-
'''
@Author:LinXinzhu
@email:18210980055@fudan.edu.cn
'''

import tensorflow as tf
import numpy as np
import os
import random
import time

class Solver(object):
    #类的初始化
    def __init__(self, model, training_iter, word2idx, idx2word, log_dir, model_save_dir):

        self.model = model
        self.training_iter = training_iter

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.log_dir = log_dir
        self.model_save_dir = model_save_dir

        self.word2idx = word2idx
        self.idx2word = idx2word
    #加载数据
    def load_data(self, split='train'):
        texts_data, labels_data,teacherlabel = [], [],[]
        if split == 'distil':
            print("USE distil data!!")
            with open('./ids/data/noaugmented.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
            with open('./data/data/noaugmented.tsv') as fin:
                count=0
                for line in fin.readlines():
                    count+=1
                    if count==1:
                        continue
                    wordlist=line.strip().split()
                    wordlist=wordlist[len(wordlist)-2:]
                    s = np.array([float(wordlist[0]),float(wordlist[1])])
                    z = np.exp(s) / sum(np.exp(s))
                    z.tolist()
                    m = z[1]
                    teacherlabel.append([float(wordlist[0]),float(wordlist[1])])
                    if m>=0.5:
                        labels_data.append(1)
                    else:
                        labels_data.append(0)
            if len(texts_data)!=len(labels_data):
                print(len(texts_data),len(labels_data))
                #防止不匹配
                raise ConnectionAbortedError
        if split == 'train':
            with open('./ids/data/sentiment.train.1.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(1)
            with open('./ids/data/sentiment.train.0.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(0)
        elif split == 'dev':
            with open('./ids/data/sentiment.dev.1.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(1)
            with open('./ids/data/sentiment.dev.0.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(0)
        elif split == 'distil_dev':
            with open('./ids/data/sentiment.dev.1.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(1)
                    teacherlabel.append([0,0])
            with open('./ids/data/sentiment.dev.0.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(0)
                    teacherlabel.append([0,0])
        elif split == 'test':
            with open('./ids/data/sentiment.test.1.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(1)
            with open('./ids/data/sentiment.test.0.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(0)

        #打乱顺序
        '''
        c=list(zip(texts_data,labels_data,teacherlabel))
        random.shuffle(c)
        texts_data,labels_data,teacherlabel = zip(*c)
        texts_data=list(texts_data)
        labels_data=list(labels_data)
        teacherlabel=list(teacherlabel)
        '''
        #变成array
        texts_data = np.array(texts_data)
        labels_data = np.array(labels_data)
        teacherlabel=np.array(teacherlabel)

        #print(texts_data.shape,labels_data.shape,teacherlabel.shape)

        shuffle_idx = np.random.permutation(range(len(texts_data)))
        texts_data = texts_data[shuffle_idx]
        labels_data = labels_data[shuffle_idx]
        if split=="distil" or split=="distil_dev":
            teacherlabel = teacherlabel[shuffle_idx]

        return texts_data, labels_data,teacherlabel
    #padding
    def prepare_text_batch(self, batch, pad_to_max=True):
        maxlen = self.model.maxlen if pad_to_max else max([len(b) for b in batch])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences=batch,
                                                               maxlen=maxlen,
                                                               padding='post',
                                                               value=0)
        return padded
    #训练（无蒸馏）
    def train(self):

        # load data
        train_texts, train_labels,_ = self.load_data('train')
        dev_texts, dev_labels,_ = self.load_data('dev')

        model = self.model

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            train_loss = 0.
            train_acc = 0.

            best_dev_acc = - np.inf
            #开始迭代
            for step in range(self.training_iter+1):
                i = step % int(len(train_texts) // model.batch_size)
                batch_texts = train_texts[i*model.batch_size:(i+1)*model.batch_size]#获得训练batch
                batch_labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]#获得训练batch

                batch_inp_text = self.prepare_text_batch(batch_texts)

                feed_dict = {model.texts: batch_inp_text,
                             model.labels: batch_labels,
                             model.keep_prob: 0.75}

                _loss, _acc, _ = sess.run([model.oldloss, model.accuracy, model.oldtrain_op], feed_dict)

                train_loss += _loss
                train_acc += _acc

                if (step+1) % 10 == 0:
                    train_loss /= 10.0
                    train_acc /= 10.0

                    print('Training[%d/%d]:  train_loss:[%.4f] train_acc:[%.4f]'
                          % (step+1, self.training_iter, train_loss, train_acc))

                    train_loss = 0.
                    train_acc = 0.

                if (step+1) % 400 == 0:
                    dev_loss = 0.
                    dev_acc = 0.

                    num_of_batch = len(dev_texts) // model.batch_size
                    time_start = time.time()
                    for j in range(num_of_batch):
                        dev_batch_texts = dev_texts[j*model.batch_size:(j+1)*model.batch_size]
                        dev_batch_labels = dev_labels[j*model.batch_size:(j+1)*model.batch_size]
                        dev_batch_inp = self.prepare_text_batch(dev_batch_texts)

                        _loss_dev, _acc_dev = sess.run([model.oldloss, model.accuracy],
                                                 feed_dict={model.texts: dev_batch_inp,
                                                            model.labels: dev_batch_labels,
                                                            model.keep_prob: 1.0})
                        dev_loss += _loss_dev
                        dev_acc += _acc_dev

                    time_end = time.time()
                    print('totally cost', time_end - time_start)
                    dev_loss /= num_of_batch
                    dev_acc /= num_of_batch

                    print('Developing[%d/%d]: dev_loss:[%.4f] dev_acc:[%.4f] best_dev_acc:[%.4f]'
                          % (step+1, self.training_iter, dev_loss, dev_acc,best_dev_acc))

                    if dev_acc > best_dev_acc:
                        print('Saving model ...')
                        saver.save(sess, os.path.join(self.model_save_dir, 'best-model'))
                        best_dev_acc = dev_acc
    #蒸馏
    def distil(self):

        # load data
        train_texts, train_labels,teacherlable = self.load_data('distil')
        dev_texts, dev_labels,dev_teacherlabel = self.load_data('distil_dev')

        model = self.model

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            wirter = tf.summary.FileWriter('logs/', sess.graph)
            train_loss = 0.
            train_acc = 0.

            best_dev_acc = - np.inf
            #开始迭代
            start_time_train = time.time()  # 开始时间
            for step in range(self.training_iter+1):
                i = step % int(len(train_texts) // model.batch_size)

                if i==int(len(train_texts) // model.batch_size):
                    end_time_train=time.time()
                    print("time:", (end_time_train - start_time_train))  # 结束时间-开始时间

                batch_texts = train_texts[i*model.batch_size:(i+1)*model.batch_size]
                batch_labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]
                batch_teacherlabel=teacherlable[i*model.batch_size:(i+1)*model.batch_size]

                batch_inp_text = self.prepare_text_batch(batch_texts)

                feed_dict = {model.texts: batch_inp_text,
                             model.labels: batch_labels,
                             model.teacherlabel:batch_teacherlabel,
                             model.keep_prob: 0.5}

                _loss, _acc, _ = sess.run([model.loss, model.accuracy, model.train_op], feed_dict)

                train_loss += _loss
                train_acc += _acc

                if (step+1) % 10 == 0:
                    train_loss /= 10.0
                    train_acc /= 10.0

                    print('Training[%d/%d]:  train_loss:[%.4f] train_acc:[%.4f]'
                          % (step+1, self.training_iter, train_loss, train_acc))

                    train_loss = 0.
                    train_acc = 0.

                if (step+1) % 400 == 0:
                    dev_loss = 0.
                    dev_acc = 0.

                    num_of_batch = len(dev_texts) // model.batch_size

                    start_time = time.time()  # 开始时间
                    for j in range(num_of_batch):
                        dev_batch_texts = dev_texts[j*model.batch_size:(j+1)*model.batch_size]
                        dev_batch_labels = dev_labels[j*model.batch_size:(j+1)*model.batch_size]
                        dev_batch_teacherlabel=dev_teacherlabel[j*model.batch_size:(j+1)*model.batch_size]

                        dev_batch_inp = self.prepare_text_batch(dev_batch_texts)

                        #print(dev_batch_labels.shape) 128*2
                        _loss_dev, _acc_dev = sess.run([model.loss, model.accuracy],
                                                 feed_dict={model.texts: dev_batch_inp,
                                                            model.labels: dev_batch_labels,
                                                            model.teacherlabel: dev_batch_teacherlabel,
                                                            model.keep_prob: 1.0})
                        dev_loss += _loss_dev
                        dev_acc += _acc_dev
                    end_time = time.time()  # 结束时间
                    print("time:",(end_time - start_time))  # 结束时间-开始时间
                    dev_loss /= num_of_batch
                    dev_acc /= num_of_batch
                    #保存最好的模型
                    if dev_acc > best_dev_acc:
                        print('Saving model ...')
                        saver.save(sess, os.path.join(self.model_save_dir, 'best-model'))
                        best_dev_acc = dev_acc
                    print('Developing[%d/%d]: dev_loss:[%.4f] dev_acc:[%.4f] best_dev_acc:[%.4f]'
                          % (step + 1, self.training_iter, dev_loss, dev_acc, best_dev_acc))
    def test(self):

        # load test dataset
        texts, labels,_ = self.load_data('test')

        model = self.model

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            restorer = tf.train.Saver()
            restorer.restore(sess, os.path.join(self.model_save_dir, 'best-model'))

            test_acc = 0.
            num_of_batch = len(texts) // model.batch_size

            for j in range(num_of_batch):
                text_batch = texts[j*model.batch_size:(j+1)*model.batch_size]
                labels_batch = labels[j*model.batch_size:(j+1)*model.batch_size]
                inp_batch = self.prepare_text_batch(text_batch)

                _acc = sess.run(model.accuracy,
                                feed_dict={model.texts: inp_batch,
                                           model.labels: labels_batch,
                                           model.keep_prob: 1.0})

                test_acc += _acc

            test_acc /= num_of_batch

            print('Test Accuracy:', test_acc)

