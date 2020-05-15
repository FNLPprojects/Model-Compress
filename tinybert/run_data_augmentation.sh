# Data augmentation aims to expand the task-specific training set.

nvidia-smi

DATA_DIR=data
BERT_BASE_DIR='bert-base-uncased'
GLOVE_EMB='./glove/glove.840B.300d.txt'
GLUE_DIR='./data'
TASK_NAME=SST-2

GPU=0
CUDA_VISIBLE_DEVICES=$GPU python data_augmentation.py \
  --pretrained_bert_model $BERT_BASE_DIR \
  --glove_embs $GLOVE_EMB \
  --glue_dir $GLUE_DIR \
  --task_name $TASK_NAME