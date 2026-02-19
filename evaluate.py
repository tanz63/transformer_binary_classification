import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE
from data import get_data_loaders
from model import TransformerClassifier


def evaluate_model(model_path='best_model.pth'):
    """评估训练好的模型"""
    print(f"Using device: {DEVICE}")
    
    # 加载数据
    _, test_loader = get_data_loaders()
    
    # 加载模型
    model = TransformerClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print(f"Model loaded from {model_path}")
    
    # 收集预测结果
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            
            outputs = model(batch_x)
            predictions = (outputs > 0.5).cpu().numpy()
            
            all_preds.extend(predictions)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算准确率
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # 详细分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved to confusion_matrix.png")
    
    return accuracy


def visualize_predictions(model_path='best_model.pth', num_samples=5):
    """可视化一些预测样本"""
    from data import generate_synthetic_data
    
    # 生成一些样本
    X, y = generate_synthetic_data(num_samples=num_samples * 2)
    
    # 加载模型
    model = TransformerClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 预测
    with torch.no_grad():
        X_device = X.to(DEVICE)
        probs = model(X_device).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    
    # 可视化
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # 类别 0 样本
        axes[0, i].plot(X[i, :, 0].numpy())
        axes[0, i].set_title(f'True: {y[i]}, Pred: {preds[i]}\nProb: {probs[i]:.3f}')
        axes[0, i].set_ylim(-3, 3)
        
        # 类别 1 样本
        idx = num_samples + i
        axes[1, i].plot(X[idx, :, 0].numpy())
        axes[1, i].set_title(f'True: {y[idx]}, Pred: {preds[idx]}\nProb: {probs[idx]:.3f}')
        axes[1, i].set_ylim(-3, 3)
    
    axes[0, 0].set_ylabel('Class 0')
    axes[1, 0].set_ylabel('Class 1')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print("Sample predictions saved to sample_predictions.png")


if __name__ == '__main__':
    import os
    
    # 评估模型
    model_path = 'best_model.pth' if os.path.exists('best_model.pth') else 'final_model.pth'
    
    if os.path.exists(model_path):
        evaluate_model(model_path)
        visualize_predictions(model_path)
    else:
        print(f"Model file not found: {model_path}")
        print("Please run train.py first.")
