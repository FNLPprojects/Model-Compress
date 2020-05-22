
# ReadMe

### 操作指南

```
python -m albert.run_classifier \
  --data_dir=albert \
  --output_dir=albert/tmp \
  --init_checkpoint=albert/albert_base/model.ckpt-best \
  --albert_config_file=albert/albert_base/albert_config.json \
  --spm_model_file=albert/albert_base/30k-clean.model \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --max_seq_length=128 \
  --optimizer=adamw \
  --task_name=SST-2 \
  --warmup_step=1000 \
  --learning_rate=3e-5 \
  --train_step=10000 \
  --save_checkpoints_steps=100 \
  --train_batch_size=128
```

### 核心代码

##### modeling

```
  if input_width != hidden_size:
    prev_output = dense_layer_2d(
        input_tensor, hidden_size, create_initializer(initializer_range),
        None, use_einsum=use_einsum, name="embedding_hidden_mapping_in")#增加了embedding_hidden_mapping_in层，即增加了E*H的矩阵
  else:
    prev_output = input_tensor
```

