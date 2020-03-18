# -*- coding:utf-8 -*-
'''
@Author:LinXinzhu
@email:18210980055@fudan.edu.cn
'''

import tensorflow as tf
from prepro import create_vocabulary, create_sst2_ids,createdata,create_vocabulary_distil, create_sst2_ids_distil
from solver import Solver
from model import BiLSTM
#设置参数
flags = tf.app.flags
flags.DEFINE_string('mode', 'distil', "train / test / distil")
flags.DEFINE_integer('vocab_size', 5000, 'size of vocab list')
flags.DEFINE_integer('batch_size', 128, 'size of batch')
flags.DEFINE_integer('num_embedding_units', 300, 'size of embeddings layer')
flags.DEFINE_integer('num_hidden_units', 300, 'size of hidden layer')
flags.DEFINE_integer('maxlen', 100, 'max length')

flags.DEFINE_integer('train_step', 50000, 'step of pretraining')
flags.DEFINE_string('model_save_dir', './save/', 'path of saving model')
flags.DEFINE_string('log_dir', './logs/', 'path of logs')

FLAGS = flags.FLAGS

if __name__ == '__main__':
    if not tf.gfile.Exists('./ids/sst2/sentiment.train.0.ids'):
        #生成指定格式数据集
        createdata()

    if FLAGS.mode != 'distil' :
        #创建词表
        word2idx, idx2word, vocab_path = create_vocabulary(FLAGS.vocab_size)
        create_sst2_ids(word2idx)
    else:
        #创建词表（增强数据集）
        word2idx, idx2word, vocab_path = create_vocabulary_distil(FLAGS.vocab_size)
        create_sst2_ids_distil(word2idx)

    if not tf.gfile.Exists(FLAGS.model_save_dir):
        tf.gfile.MakeDirs(FLAGS.model_save_dir)
    #创建模型对象
    model = BiLSTM(vocab_size=FLAGS.vocab_size,
                   batch_size=FLAGS.batch_size,
                   embedding_size=FLAGS.num_embedding_units,
                   num_hidden_size=FLAGS.num_hidden_units,
                   maxlen=FLAGS.maxlen)
    #创建训练对象
    solver = Solver(model=model,
                    training_iter=FLAGS.train_step,
                    word2idx=word2idx,
                    idx2word=idx2word,
                    log_dir=FLAGS.log_dir,
                    model_save_dir=FLAGS.model_save_dir)

    if FLAGS.mode == 'train':
        solver.train()
    elif FLAGS.mode == 'test':
        solver.test()
    elif FLAGS.mode=='distil':
        solver.distil()

