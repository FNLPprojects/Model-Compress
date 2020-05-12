# ReadMe

### distilLSTM

#### **实验环境**

基于python3.7+tensorflow1.15 

GPU：GTX2080TI

**Bert模型下载**

```shell
cd bert
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip 
```

#### **Bert模型微调**

```shell
python run_classifier.py \
 --task_name=SST2 \
 --do_train=true \
 --do_eval=true \
 --data_dir=SST-2 \
 --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
 --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
 --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
 --max_seq_length=128 \
 --train_batch_size=32 \
 --learning_rate=2e-5 \
 --num_train_epochs=3.0 \
 --output_dir=/tmp/SST2_output/
```

**参数选项**

--task_name 指定数据集任务

 --do_train 启用训练      

 --do_eval 启用测试

 --data_dir 设置微调的数据集

 --vocab_file Bert词表位置

 --bert_config_file Bert参数文件位置

 --init_checkpoint Bert模型位置

 --max_seq_length 最大句子长度

 --train_batch_size 设置batch size大小，Bert-base-uncased设置为32需要11G显存空间

 --learning_rate 设置学习率

 --num_train_epochs 设置训练迭代次数

 --output_dir 设置微调之后的模型输出文件夹

#### **创建数据增强的数据集**
默认给出了SST2的增强数据集，可跳过
```shell
python generate_dataset.py
```

#### **蒸馏模型**

```shell
cd distilLSTM
python main.py
```

**参数选项**

```shell
--mode 设置训练/测试/蒸馏（train/test/distil）
```

**主要函数说明**

1. prepro.py

- creat_vocabulary：创建word2idx和idx2word，在使用其他数据集时需要修改这个函数
- creat_sst2_ids：读取原始sst2文件生成ids文件，在使用其他数据集时需要修改这个函数

2. model.BiLSTM

- build_model：构建具有带训练参数的模型节点，在使用其他Student Model时需要替换该模块
- build_graph：构建计算图

3. solver.Solver

- load_data: 从ids文件中读取数据，当训练模式改为蒸馏（distil）时，会从增强数据集中读取数据，并将Bert的logtic 输出转化为np.array
- prepare_text_batch: 将长短不一的输入文本padding为相同长度的输入

train: 使用train数据集训练模型，根据模型在dev集上的表现保存最优模型

test: 测试模型在test数据集上的准确率

distill:使用增强数据集蒸馏模型，根据模型在dev集上的表现保存最优模型
