# TinyBERT
["TinyBERT: Distilling BERT for Natural Language Understanding"](https://arxiv.org/abs/1909.10351)论文的实现

## 依赖
- Python 3.6
- PyTorch 1.4.0
- transformers 2.9.0

完整的环境可以通过以下命令安装：
```bash
conda create -n tinybert python=3.6
```

```bash
pip install -r requirements.txt
```
      
## 通用蒸馏 (General Distillation)
通用蒸馏阶段使用预训练得到的 BERT-base 为教师模型，在大规模文本语料上进行知识蒸馏得到通用的TinyBERT。
这个操作可以让TinyBERT学习到通用的语义表示，提高了模型的泛化能力，也为随后针对特定任务的蒸馏提供了一个很好的初始化。

通用蒸馏包含两步：
（1）语料预处理 （2）进行通用蒸馏

### 1. 语料预处理
准备大规模语料，比如[WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)。可以用过如下命令下载：
```
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
rm wikitext-103-raw-v1.zip
```
执行以下命令，进行训练数据预处理
- CORPUS_RAW：大规模语料，比如说Wikipedia
- BERT_BASE_DIR：教师模型类型
- OUTPUT_DIR: 处理过的语料保存路径
直接执行
```bash
bash run_pregenerate_training_data.sh
```
或者 执行
```bash
CORPUS_RAW='./data/wikitext-103-raw/wiki.train.raw'
BERT_BASE_DIR='bert-base-uncased'
OUTPUT_DIR='./data/pretrain_data_json'

python pregenerate_training_data.py \
  --train_corpus $CORPUS_RAW \
  --bert_model $BERT_BASE_DIR \
  --do_lower_case \
  --epochs_to_generate 3 \
  --output_dir $OUTPUT_DIR
```

### 2. 执行通用蒸馏
执行以下命令，进行通用蒸馏
- CORPUS_JSON: 预处理过的语料
- STUDENT_CONFIG_DIR: 学生模型的配置文件 (config)
- GENERAL_TINYBERT_DIR: 保存路径
直接执行
```bash
bash run_general_student_distil.sh
```
或者 执行
```bash
CORPUS_JSON='./data/pretrain_data_json'
STUDENT_CONFIG_DIR='student_model/student_4L_312D_config.json' 
#GENERAL_TINYBERT_DIR='student_model/General_TinyBERT_4L321D'
GENERAL_TINYBERT_DIR='student_model/General_TinyBERT_xLxD'

python general_student.py \
  --pregenerated_data $CORPUS_JSON \
  --teacher_model 'bert-base-uncased' \
  --student_model $STUDENT_CONFIG_DIR \
  --do_lower_case \
  --train_batch_size 32 \
  --output_dir $GENERAL_TINYBERT_DIR
```

经过了通用蒸馏得到了两种规模的通用TinyBERT： General_TinyBERT(4layer-312dim)和General_TinyBERT(6layer-768dim)

下载地址如下：
链接: https://pan.baidu.com/s/1HkuH9CakQqFXpbpuHaPB0g 
提取码: uq85 

可以将他们下载下来，放置在'./student_model'路径下，如'./student_model/General_TinyBERT_3L_312D'和'./student_model/General_TinyBERT_6L_768D'


## 数据增强
数据增强是TinyBERT中重要的一步个步骤，通过数据增强步骤，TinyBERT可以学习更多的任务相关的例子，可以进一步提高学生模型的泛化能力。可以帮助TinyBERT获得和BERT-base相匹配的性能，甚至在部分任务上超过BERT-base的表现。

### 1. GLUE数据集下载
可以通过执行以下脚本下载GLUE任务的所有数据集，将会自动下载并解压到'--data_dir=data'目录下。

```
bash run_download_glue_data.sh
```
或者
```bash
python download_glue_data.py --data_dir data --tasks all
```

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "diagnostic"]

以上脚本将会默认下载所有BLUE任务数据集，也可以通过'--tasks=TASKS'，指定下载某些数据集

### 2. 下载GloVe嵌入
TinyBERT所采用的数据增强方法，结合了预训练BERT和GloVe嵌入来做词级别的替换。
可以同以下脚本下载GloVe嵌入，放置到'./glove'目录下
```
cd glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip 
rm glove.840B.300d.zip 
```

### 3. 进行数据增强
通过执行以下脚本进行数据增强
``` bash
bash run_data_augmentation.sh
```
增强后的数据集 train_aug.tsv 会自动保存到相应的GLUE任务数据集下。


### 任务特定蒸馏 (Task-specific Distillation)
在任务特定蒸馏中，将重新对得到的通用TinyBERT进行微调。通过在特定任务上进行微调，来进一步改进TinyBERT。任务特定化蒸馏包括三个步骤:
（1）微调教师BERT，随后（2）微调学生TinyBERT，包含中间层蒸馏和预测层蒸馏。

### 1. 微调教师模型BERT
执行以下脚本进行微调教师模型
- DATA_DIR: GLUE数据集总路径
- TASK_NAME: 任务名
- OUTPUT_DIR: 模型保存路径

直接执行
```bash
bash run_train_task_teacher.sh
```
或者
```bash
DATA_DIR=data
TASK_NAME=SST-2
OUTPUT_DIR="finetuned_teacher/$TASK_NAME/"

python task_teacher.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --data_dir "$DATA_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 50 \
  --num_train_epochs 3.0 \
  --output_dir $OUTPUT_DIR \
  --save_steps 1000 \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_lower_case
```
### 2. 微调学生模型TinyBERT
#### (1) 中间层蒸馏
执行以下命令进行中间层蒸馏，
- DATA_DIR: GLUE数据集总路径
- TASK_NAME: 任务名
- FT_BERT_BASE_DIR: 在特定任务上微调过的教师模型路径
- GENERAL_TINYBERT_DIR: 通用TinyBERT模型路径
- TMP_TINYBERT_DIR: 临时TinyBERT模型路径

直接执行
```bash
bash run_task_student_intermediate_distil.sh
```
或者
```bash
DATA_DIR='./data'
TASK_NAME=SST-2
TASK_DIR=$DATA_DIR/$TASK_NAME
FT_BERT_BASE_DIR="./finetuned_teacher/$TASK_NAME/checkpoint-6000"
GENERAL_TINYBERT_DIR='./student_model/General_TinyBERT_4L_312D'
TMP_TINYBERT_DIR="./student_model/$TASK_NAME/tmp_tinybert"

python task_student.py \
  --teacher_model $FT_BERT_BASE_DIR \
  --student_model $GENERAL_TINYBERT_DIR \
  --data_dir $TASK_DIR \
  --task_name $TASK_NAME \
  --output_dir $TMP_TINYBERT_DIR \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --num_train_epochs 10 \
  --aug_train \
  --do_lower_case
```

### (2) 预测层蒸馏
执行以下命令进行预测层蒸馏，
- DATA_DIR: GLUE数据集总路径
- TASK_NAME: 任务名
- FT_BERT_BASE_DIR: 在特定任务上微调过的教师模型路径
- TMP_TINYBERT_DIR: 临时TinyBERT模型路径
- TINYBERT_DIR: 任务特定TinyBERT模型路径

直接执行
```bash
bash run_task_student_prediction_distil.sh
```
或者
```bash
DATA_DIR='./data'
TASK_NAME=SST-2
TASK_DIR=$DATA_DIR/$TASK_NAME
FT_BERT_BASE_DIR="./finetuned_teacher/$TASK_NAME/checkpoint-6000"
TMP_TINYBERT_DIR="./student_model/$TASK_NAME/tmp_tinybert"
TINYBERT_DIR="./student_model/$TASK_NAME/tinybert"

python task_student.py \
  --pred_distill  \
  --teacher_model $FT_BERT_BASE_DIR \
  --student_model $TMP_TINYBERT_DIR \
  --data_dir $TASK_DIR \
  --task_name $TASK_NAME \
  --output_dir $TINYBERT_DIR \
  --aug_train  \
  --do_lower_case \
  --learning_rate 3e-5  \
  --num_train_epochs  3  \
  --max_seq_length 128 \
  --train_batch_size 32
```

### 3. 性能测试
通过执行以下脚本，在GLUE任务上进行性能测试：
- DATA_DIR: GLUE数据集总路径
- TASK_NAME: 任务名
- TINYBERT_DIR: 任务特定TinyBERT模型路径
- OUTPUT_DIR：测试结果保存路径

直接执行
```bash
bash run_task_eval.sh
```
或者
```bash
DATA_DIR='./data'
TASK_NAME=SST-2
TASK_DIR=$DATA_DIR/$TASK_NAME
TINYBERT_DIR="./student_model/$TASK_NAME/tinybert"
OUTPUT_DIR="./output"

python task_student.py \
  --do_eval \
  --student_model $TINYBERT_DIR \
  --data_dir $TASK_DIR \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --do_lower_case \
  --eval_batch_size 32 \
  --max_seq_length 128
```