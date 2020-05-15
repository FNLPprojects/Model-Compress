nvidia-smi

CORPUS_JSON='./data/pretrain_data_json'
STUDENT_CONFIG_DIR='student_model/student_4L_312D_config.json' # includes the config file of student_model.
#GENERAL_TINYBERT_DIR='student_model/General_TinyBERT_4L321D'
GENERAL_TINYBERT_DIR='student_model/test'

# train_batch_size 256
GPU=0
CUDA_VISIBLE_DEVICES=$GPU python general_student.py \
  --pregenerated_data $CORPUS_JSON \
  --teacher_model 'bert-base-uncased' \
  --student_model $STUDENT_CONFIG_DIR \
  --do_lower_case \
  --train_batch_size 32 \
  --output_dir $GENERAL_TINYBERT_DIR