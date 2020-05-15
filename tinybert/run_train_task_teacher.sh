nvidia-smi

DATA_DIR=data
TASK_NAME=SST-2
OUTPUT_DIR="finetuned_teacher/$TASK_NAME/"
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python task_teacher.py \
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