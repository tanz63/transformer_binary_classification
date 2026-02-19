"""
工具函数和类定义
包含学习率调度器等训练辅助工具
"""

import torch
import math
import matplotlib.pyplot as plt
from typing import List, Optional


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    带 warmup 的余弦退火学习率调度器
    
    该调度器结合了 warmup 阶段和余弦退火阶段：
    1. Warmup 阶段：学习率从 0 线性增加到初始学习率
    2. 余弦退火阶段：学习率按照余弦函数从初始学习率衰减到最小学习率
    
    参数:
        optimizer (torch.optim.Optimizer): 要调整学习率的优化器
        warmup_epochs (int): warmup 阶段的 epoch 数
        total_epochs (int): 总训练 epoch 数
        min_lr (float, optional): 最小学习率，默认为初始学习率的 1/100
        last_epoch (int, optional): 最后一个 epoch 的索引，默认为 -1
        verbose (bool, optional): 是否打印更新信息，默认为 False
    
    属性:
        warmup_epochs (int): warmup 阶段的 epoch 数
        total_epochs (int): 总训练 epoch 数
        min_lr (float): 最小学习率
        base_lrs (List[float]): 优化器的初始学习率列表
    
    示例:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, total_epochs=20)
        >>> for epoch in range(20):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 warmup_epochs: int, 
                 total_epochs: int,
                 min_lr: Optional[float] = None,
                 last_epoch: int = -1,
                 verbose: bool = False):
        
        # 参数验证
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs 必须为非负整数，当前为 {warmup_epochs}")
        if total_epochs <= 0:
            raise ValueError(f"total_epochs 必须为正整数，当前为 {total_epochs}")
        if warmup_epochs >= total_epochs:
            raise ValueError(f"warmup_epochs ({warmup_epochs}) 必须小于 total_epochs ({total_epochs})")
        
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # 如果没有指定最小学习率，则使用初始学习率的 1/100
        if min_lr is None:
            self.min_lr = [base_lr / 100.0 for base_lr in optimizer.defaults['lr']]
        else:
            self.min_lr = [min_lr for _ in optimizer.param_groups]
        
        # 保存优化器的初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """
        计算当前 epoch 的学习率
        
        返回:
            List[float]: 每个参数组的学习率列表
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加学习率
            progress = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            # 确保 progress 在 [0, 1] 范围内
            progress = min(1.0, max(0.0, progress))
            
            # 余弦退火公式：lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
            return [
                min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr, min_lr in zip(self.base_lrs, self.min_lr)
            ]
    
    def plot_schedule(self, save_path: Optional[str] = None) -> None:
        """
        绘制学习率调度曲线
        
        参数:
            save_path (str, optional): 保存图像的路径，如果不提供则显示图像
        
        返回:
            None
        """
        # 模拟所有 epoch 的学习率
        lr_history = []
        for epoch in range(self.total_epochs):
            self.last_epoch = epoch
            lr_history.append(self.get_lr()[0])  # 只取第一个参数组的学习率
        
        # 重置 last_epoch
        self.last_epoch = -1
        
        # 绘制图像
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.total_epochs), lr_history, 'b-', linewidth=2)
        plt.axvline(x=self.warmup_epochs - 0.5, color='r', linestyle='--', alpha=0.7, label='Warmup 结束')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'CosineWarmupScheduler 学习率变化曲线\n(Warmup: {self.warmup_epochs} epochs, Total: {self.total_epochs} epochs)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"学习率调度曲线已保存到: {save_path}")
        else:
            plt.show()


def test_scheduler():
    """
    测试 CosineWarmupScheduler 的功能
    
    这个函数展示了如何使用调度器并可视化学习率变化
    """
    # 创建一个简单的模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建调度器
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        total_epochs=20,
        min_lr=1e-5
    )
    
    print("CosineWarmupScheduler 测试")
    print(f"初始学习率: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Warmup epochs: {scheduler.warmup_epochs}")
    print(f"Total epochs: {scheduler.total_epochs}")
    print(f"最小学习率: {scheduler.min_lr[0]:.6f}")
    print("\n学习率变化:")
    
    # 模拟训练过程并记录学习率
    lr_history = []
    for epoch in range(scheduler.total_epochs):
        lr = optimizer.param_groups[0]['lr']
        lr_history.append(lr)
        
        if epoch < 5 or epoch % 5 == 0 or epoch == scheduler.total_epochs - 1:
            print(f"Epoch {epoch:3d}: 学习率 = {lr:.6f}")
        
        # 更新学习率（模拟训练步骤）
        scheduler.step()
    
    # 绘制学习率曲线
    scheduler.plot_schedule('cosine_warmup_schedule.png')
    
    return scheduler, lr_history


if __name__ == '__main__':
    # 如果直接运行此文件，则执行测试
    scheduler, lr_history = test_scheduler()
    print("\n测试完成！学习率调度曲线已保存为 'cosine_warmup_schedule.png'")