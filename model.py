import torch
import torch.nn as nn
import math
from config import D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT


class TransformerClassifier(nn.Module):
    """
    基于 Transformer Encoder 的二分类模型
    """
    
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, 1) - 分类概率
        """
        # Transformer 编码
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 全局平均池化
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # 分类
        output = self.classifier(pooled)  # (batch, 1)
        
        return output.squeeze(-1)  # (batch,)


if __name__ == '__main__':
    # 测试模型
    model = TransformerClassifier()
    
    # 模拟输入
    batch_size, seq_len, d_model = 4, 50, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
