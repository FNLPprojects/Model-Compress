# PKD for BERT Model Compression
参考论文 ["Patient Knowledge Distillation for BERT Model Compression"](https://arxiv.org/abs/1908.09355).

### Requirement
python == 3.6 \
torch == 1.4.0 \
transformers

### Compress a BERT
1. 将对应的数据集名称放入data文件夹 如data/SST-2
2. 将预训练的bert模型(bert-base-uncased)文件夹放在主目录下
3. 指定数据路径和任务名 执行以下命令进行教师模型的微调。举例如下

```bash
export TASK_NAME=SST-2

python main.py \
  --task_name $TASK_NAME \
  --data_dir data/$TASK_NAME \
  --output_dir teacher_output/$TASK_NAME \
  --train_type finetune_teacher \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --num_train_epochs 3
```

4. 执行以下命令进行学生模型的训练

```bash
python main.py \
  --task_name $TASK_NAME \
  --data_dir data/$TASK_NAME \
  --output_dir student_output/$TASK_NAME \
  --train_type train_student \
  --do_train \
  --do_eval \
  --num_train_epochs 4 \
  --teacher_prediction teacher_output/$TASK_NAME/sst-2_teacher_12layer_information.pkl \
  --overwrite_output_dir \
  --encoder_checkpoint teacher_output/$TASK_NAME/finetune_teacher_epoch2.encoder.pkl \
  --cls_checkpoint teacher_output/$TASK_NAME/finetune_teacher_epoch2.cls.pkl
```

### 主要函数说明
1. src/data_processor.py \
get_task_dataloader 读取数据集（学生模型的训练时会加载所需要的知识）

2. src/KD_loss.py \
distillation_loss 包含交叉熵和KL散度的损失函数 \
patience_loss 衡量[CLS]差距的MSE损失函数 

3. src/model.py \
BertForSequenceClassificationEncoder  Bert模型的编码层 \
FCClassifierForSequenceClassification Bert模型的分类层

4. src/utils.py \
compute_metrics 评测准确率 \
count_parameters 计算模型参数量 \
load_model 加载模型  model的含义 exact:完全加载 student:加载编码部分 教师12层 学生6层 classifier:加载分类层 \
eval_model_dataloader 记录教师模型的信息 包括损失、准确率、logit输出和对应的[CLS]层信息

5. main.py \
train 训练模型 \
evaluate 评测模型

### 参数选项
--task_name 任务名称 \
--data_dir 数据集文件夹 \
--output_dir 输出文件夹 \
--bert_model 使用的bert模型 \
--encoder_checkpoint 编码层的保存文件（学生模型用） \
--cls_checkpoint 分类层的保存文件（学生模型用) \
--alpha 交叉熵和KL散度的权重 \
--T 蒸馏温度 \
--beta patient损失的权重 \
--fc_layer_idx 学生模型与教师模型的对应层数（默认设置为1,3,5,7,9,最后一层不用训练调整参数） \
--normalize_patience 是否标准化patient损失 \
--do_train 训练 \
--do_eval 测试 \
--train_type finetune_teacher:微调教师模型 train_student:训练学生模型 \
--max_seq_length 输入的句子最大长度 \
--seed 初始化随机种子 \
--train_batch_size 训练时的batch大小 \
--eval_batch_size 测试时的batch大小 \
--learning_rate 学习率 \
--adam_epsilon Adam优化器的模糊因子 \
--num_train_epochs 训练的epoch数 \
--gradient_accumulation_steps 梯度累计的步数 \
--logging_steps 每多少步输出一次结果 \
--student_hidden_layers 模型的transformer层数 教师为12 学生为6 \
--teacher_prediction 训练学生模型时指定的包含教师预测信息的文件 \
--warmup_steps 预热步数 \
--overwrite_output_dir 是否覆盖已有的输出信息


