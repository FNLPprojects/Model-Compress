
nvidia-smi

DATA_DIR='./data'
TASK_NAME=SST-2
TASK_DIR=$DATA_DIR/$TASK_NAME
FT_BERT_BASE_DIR="./finetuned_teacher/$TASK_NAME/checkpoint-6000"
TMP_TINYBERT_DIR="./student_model/$TASK_NAME/tmp_tinybert"
TINYBERT_DIR="./student_model/$TASK_NAME/tinybert"

GPU=2
CUDA_VISIBLE_DEVICES=$GPU python task_student.py \
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