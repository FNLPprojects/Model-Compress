
nvidia-smi

DATA_DIR='./data'
TASK_NAME=SST-2
TASK_DIR=$DATA_DIR/$TASK_NAME
TINYBERT_DIR="./student_model/$TASK_NAME/tinybert"
OUTPUT_DIR="./output"
GPU=3

CUDA_VISIBLE_DEVICES=$GPU python task_student.py \
  --do_eval \
  --student_model $TINYBERT_DIR \
  --data_dir $TASK_DIR \
  --task_name $TASK_NAME \
  --output_dir $OUTPUT_DIR \
  --do_lower_case \
  --eval_batch_size 32 \
  --max_seq_length 128