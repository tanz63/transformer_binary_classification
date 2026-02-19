# Transformer Binary Classification

基于 Transformer 的二分类神经网络实验项目。

## 环境配置

### Conda 环境

```bash
# 创建新环境
conda create -n transformer_clf python=3.10
conda activate transformer_clf

# 安装依赖
conda install pytorch torchvision torchaudio -c pytorch
pip install numpy matplotlib scikit-learn tqdm
```

### 已安装环境信息

- **Python**: 3.10+
- **PyTorch**: 2.x
- **NumPy**: 1.24+
- **scikit-learn**: 1.3+

## 项目结构

```
transformer_binary_classification/
├── data/               # 数据目录
├── models/             # 模型定义
├── utils/              # 工具函数
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── config.py           # 配置文件
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

## 使用方法

```bash
# 激活环境
conda activate transformer_clf

# 训练模型
python train.py

# 评估模型
python evaluate.py
```

## 模型架构

- **Input**: 序列数据 (batch_size, seq_len, d_model)
- **Transformer Encoder**: 2 层，8 头注意力
- **Classification Head**: 全局平均池化 + Linear
- **Output**: 二分类概率 (sigmoid)

## Git 仓库

GitHub: https://github.com/tanz63/transformer_binary_classification
