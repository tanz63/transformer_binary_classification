import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import SEQ_LEN, D_MODEL, NUM_SAMPLES, SEED, BATCH_SIZE


def set_seed(seed=SEED):
    """设置随机种子保证可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN, d_model=D_MODEL):
    """
    生成模拟二分类数据
    
    类别 0: 低频信号 + 噪声
    类别 1: 高频信号 + 噪声
    """
    set_seed()
    
    X = []
    y = []
    
    for i in range(num_samples):
        label = i % 2  # 交替生成两类
        
        # 生成基础信号
        t = np.linspace(0, 4 * np.pi, seq_len)
        
        if label == 0:
            # 低频信号
            signal = np.sin(t) + 0.5 * np.sin(2 * t)
        else:
            # 高频信号
            signal = np.sin(3 * t) + 0.5 * np.sin(6 * t)
        
        # 添加噪声并扩展到 d_model 维度
        noise = np.random.randn(seq_len, d_model) * 0.3
        for j in range(d_model):
            noise[:, j] += signal * (1 + 0.1 * np.random.randn())
        
        X.append(noise)
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # 转换为 PyTorch tensor
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    return X, y


class BinaryClassificationDataset(Dataset):
    """二分类数据集"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_data_loaders(test_split=0.2):
    """获取训练和测试数据加载器"""
    X, y = generate_synthetic_data()
    
    # 划分训练集和测试集
    dataset = BinaryClassificationDataset(X, y)
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader


if __name__ == '__main__':
    # 测试数据生成
    X, y = generate_synthetic_data(num_samples=10)
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Sample labels: {y}")
    
    train_loader, test_loader = get_data_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
