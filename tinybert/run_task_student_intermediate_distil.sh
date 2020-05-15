
nvidia-smi

DATA_DIR='./data'
TASK_NAME=SST-2
TASK_DIR=$DATA_DIR/$TASK_NAME
FT_BERT_BASE_DIR="./finetuned_teacher/$TASK_NAME/checkpoint-6000"
GENERAL_TINYBERT_DIR='./student_model/General_TinyBERT_4L_312D'
TMP_TINYBERT_DIR="./student_model/$TASK_NAME/tmp_tinybert"

GPU=1
CUDA_VISIBLE_DEVICES=$GPU python task_student.py \
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