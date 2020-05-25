# ReadMe

### Discrimination-aware Channel Pruning for Deep Neural Networks

**实验环境**

基于python3.6+tensorflow 1.10.1

GPU：GTX2080TI

**使用方法**

## 预训练

```shell
python main.py --data_dir_local DATA/cifar10
```

## 剪枝

```shell
python main.py --learner dis-chn-pruned --dcp_prune_ratio 0.90 --data_dir_local DATA/cifar10
```


