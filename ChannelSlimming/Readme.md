# ReadMe

### Learning Efficient Convolutional Networks Through Network Slimming (ICCV2017)

**实验环境**

基于python3.6+pytorch 1.5.0

GPU：GTX2080TI

**使用方法**

## 基线 

```shell
python main.py
```

## 稀疏化训练

```shell
python main.py -sr --s 0.0001
```

## 剪枝

```shell
python prune.py --model model_best.pth.tar --save pruned.pth.tar --percent 0.7
```

## 微调

```shell
python main.py --refine pruned.pth.tar --epochs 40
```


