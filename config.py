# 项目配置

import torch

# 数据配置
SEQ_LEN = 50          # 序列长度
D_MODEL = 64          # 特征维度
NUM_SAMPLES = 2000    # 样本数量

# 模型配置
NHEAD = 8             # 注意力头数
NUM_LAYERS = 2        # Transformer 层数
DIM_FEEDFORWARD = 256 # 前馈网络维度
DROPOUT = 0.1         # Dropout 率

# 训练配置
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机种子
SEED = 42
