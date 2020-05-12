# -*- coding:utf-8 -*-
'''
@Author:LinXinzhu
@email:18210980055@fudan.edu.cn
'''

import tensorflow as tf

class BiLSTM(object):

    def __init__(self,
                 vocab_size,
                 batch_size,
                 embedding_size,
                 num_hidden_size,
                 maxlen,
                 num_categories=2):

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_hidden_size = num_hidden_size
        self.maxlen = maxlen
        self.num_categories = num_categories

        self._build_model()
        self._build_graph()
    #构建模型
    def _build_model(self):

        with tf.device('/cpu:0'):
            with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
                self.embeddings = tf.get_variable('embedding_lookup',
                                                  [self.vocab_size, self.embedding_size],
                                                  dtype=tf.float32)

        self.hidden_proj = tf.layers.Dense(self.num_hidden_size, activation='linear')

        self.fw_encoder_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden_size, name='fw_cell')
        self.bw_encoder_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden_size, name='bw_cell')
        #全连接层
        self.discriminator_dense = tf.layers.Dense(self.num_hidden_size, name='discriminator_dense')
        self.discriminator_out = tf.layers.Dense(self.num_categories, name='discriminator_out')

    def _build_graph(self):

        self.texts = tf.placeholder(tf.int32, [None, self.maxlen], name='input_texts')
        self.labels = tf.placeholder(tf.int64, [None], name='input_labels')#label是一个一维tensor
        self.teacherlabel=tf.placeholder(tf.float32, [None,2], name='teacher_labels')#label是一个二维tensor，例如128*2

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')#dropout参数

        self.text_lens = tf.cast(tf.reduce_sum(tf.sign(self.texts), 1), tf.int32)

        text_embedding = tf.nn.embedding_lookup(self.embeddings, self.texts)
        proj_emb = self.hidden_proj(text_embedding)

        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_encoder_cell,
                                                          cell_bw=self.bw_encoder_cell,
                                                          sequence_length=self.text_lens,
                                                          inputs=proj_emb,
                                                          dtype=tf.float32)
        concat_states = tf.concat(states, 1)        # 拼接forward和backward 维度变化：2*(batch_size x hidden_size) ==> batch_size x (2*hidden_size)
        concat_states = tf.nn.dropout(concat_states, keep_prob=self.keep_prob)#设置droupout参数

        output = self.discriminator_dense(concat_states)
        output = tf.nn.tanh(output)
        self.output = self.discriminator_out(output)#最终的logtic输出

        self.ypred_for_auc = tf.nn.softmax(self.output)#经过softmax就变成了预测label


        #train时的损失函数
        self.oldloss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output+1e-10,
                labels=self.labels
            )
        )


        #损失函数logtic output 和 bert_output(teacher label)求MSE以及和真实标签做交叉熵
        alpha=0
        self.loss = alpha*tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output+1e-10,
                labels=self.labels
            )
        )+(1-alpha)*tf.reduce_mean(
            tf.losses.mean_squared_error(
                self.output + 1e-10,
                self.teacherlabel
            )
        )
        #ypred为预测标签，output先求argmax获取最大值下标，tf.cast将数据转型int64
        ypred = tf.cast(tf.argmax(self.output, 1), tf.int64)
        #判断两个tensor数值是否对应相等，求出准确的样本数
        correct = tf.equal(ypred, self.labels)
        #求出准确率
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        #设置训练变量
        tvars = tf.trainable_variables()
        #设置优化器为Adam
        opt = tf.train.AdamOptimizer()
        #设置目标优化函数
        self.train_op = opt.minimize(self.loss, var_list=tvars)
        #设置无蒸馏的目标优化
        self.oldtrain_op = opt.minimize(self.oldloss, var_list=tvars)