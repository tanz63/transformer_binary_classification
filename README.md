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

### 已安装环境信息（测试环境：pytorch_ts）

- **Python**: 3.10+
- **PyTorch**: 2.10.0
- **NumPy**: 1.24+
- **scikit-learn**: 1.3+
- **tqdm**: 4.65+
- **seaborn**: 0.12+
- **Device**: CPU (CUDA not available)

### Conda 环境路径

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_ts
```

## 项目结构

```
transformer_binary_classification/
├── data/               # 数据目录
├── models/             # 模型定义
├── utils/              # 工具函数
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── config.py           # 配置文件
├── model.py            # 模型架构
├── data.py             # 数据生成
├── utils.py            # 学习率调度器等工具
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

# 测试学习率调度器
python utils.py
```

## 学习率调度器

项目包含 `CosineWarmupScheduler` 学习率调度器（`utils.py`）：

- **Warmup 阶段**：学习率从 0 线性增加到初始学习率
- **余弦退火阶段**：学习率按余弦函数衰减到最小学习率

```python
from utils import CosineWarmupScheduler

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineWarmupScheduler(
    optimizer, 
    warmup_epochs=5, 
    total_epochs=20,
    min_lr=1e-5
)

for epoch in range(20):
    train(...)
    scheduler.step()
```

## 模型架构

- **Input**: 序列数据 (batch_size, seq_len, d_model)
- **Transformer Encoder**: 2 层，8 头注意力
- **Classification Head**: 全局平均池化 + Linear
- **Output**: 二分类概率 (sigmoid)

## Git 仓库

GitHub: https://github.com/tanz63/transformer_binary_classification
