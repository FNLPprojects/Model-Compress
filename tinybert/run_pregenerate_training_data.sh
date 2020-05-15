nvidia-smi

CORPUS_RAW='./data/wikitext-103-raw/wiki.train.raw'
BERT_BASE_DIR='bert-base-uncased' # the BERT-base teacher model
OUTPUT_DIR='./data/pretrain_data_json'
GPU=0

CUDA_VISIBLE_DEVICES=$GPU python pregenerate_training_data.py \
  --train_corpus $CORPUS_RAW \
  --bert_model $BERT_BASE_DIR \
  --do_lower_case \
  --epochs_to_generate 3 \
  --output_dir $OUTPUT_DIR

