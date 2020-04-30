# BERT-of-Theseus
参考论文 ["BERT-of-Theseus: Compressing BERT by Progressive Module Replacing"](http://arxiv.org/abs/2002.02925).

### Requirement
python == 3.6
torch == 1.4.0
transformers

### Compress a BERT
1. 将对应的数据集名称放入data文件夹 如data/SST-2。
2. 指定数据路径和任务名 执行以下命令进行教师模型的微调。举例如下

```bash
# For compression with a replacement scheduler
export DATA_DIR=data
export TASK_NAME=SST-2

python main.py \
  --train_type finetune_teacher \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --data_dir "$DATA_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 50 \
  --num_train_epochs 3.0 \
  --output_dir finetuned_teacher/$TASK_NAME/ \
  --cache_dir pretrained_model/.bert-base-uncased \
  --save_steps 1000
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_lower_case
```

3. 执行以下命令进行学生模型的训练

```bash
# For compression with a constant replacing rate
python main.py \
  --train_type student \
  --model_name_or_path finetuned_teacher/$TASK_NAME/checkpoint-6000 \
  --task_name $TASK_NAME \
  --data_dir "$DATA_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 5000 \
  --num_train_epochs 5 \
  --output_dir student_model/$TASK_NAME/ \
  --evaluate_during_training \
  --replacing_rate 0.7 \
  --steps_for_replacing 3000 \
  --do_train \
  --do_eval \
  --do_lower_case
```

### note
区别教师模型和学生模型的主要为train_type参数，代码基于huggingface的run_glue.py改写。


