#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析MobileNetV3模型的混淆矩阵，可视化分类错误分布
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"D:\python_project\BS\model_checkpoints\new_models"
TEST_LABELS_PATH = r"E:\Dataset\test_labels.csv"
BASE_DIR = r"E:\Dataset"
OUTPUT_DIR = r"D:\python_project\BS\model_checkpoints\analysis"

# 类别映射
CLASS_NAMES = {
    0: "ADI",
    1: "BACK",
    2: "DEB",
    3: "LYM",
    4: "MUC",
    5: "MUS",
    6: "NORM",
    7: "STR",
    8: "TUM"
}

# 反向映射：字符串标签到整数
STRING_TO_CLASS = {
    "ADI": 0,
    "BACK": 1,
    "DEB": 2,
    "LYM": 3,
    "MUC": 4,
    "MUS": 5,
    "NORM": 6,
    "STR": 7,
    "TUM": 8
}

# 模型定义
class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# 数据集类
class PathologicalDataset(Dataset):
    def __init__(self, labels_path, base_dir, transform=None):
        self.df = pd.read_csv(labels_path)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label_str = self.df.iloc[idx, 1]

        # 将字符串标签转换为整数
        if isinstance(label_str, str):
            label = STRING_TO_CLASS[label_str.upper()]
        else:
            label = int(label_str)

        # 拼接完整路径
        full_img_path = os.path.join(self.base_dir, img_path)

        # 读取图像
        image = Image.open(full_img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

# 加载模型
def load_model():
    """加载MobileNetV3模型"""
    model = MobileNetV3Model().to(DEVICE)
    model_path = os.path.join(MODEL_DIR, "mobilenetv3_best_20260414_195234.pth")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# 生成混淆矩阵
def generate_confusion_matrix(model, test_loader):
    """生成混淆矩阵"""
    num_classes = 9
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 填充混淆矩阵
            for true_label, pred_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix

# 绘制混淆矩阵
def plot_confusion_matrix(confusion_matrix):
    """绘制混淆矩阵"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建混淆矩阵热力图
    plt.figure(figsize=(12, 10))

    # 生成类别名称
    class_names = [CLASS_NAMES[i] for i in range(9)]

    # 计算准确率
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # 绘制热力图
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'fontsize': 18})

    # 调整右侧颜色条字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    plt.title(f'MobileNetV3模型混淆矩阵\n准确率: {accuracy:.4f}', fontsize=20, fontweight='bold')
    plt.xlabel('预测标签', fontsize=18)
    plt.ylabel('真实标签', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=16)
    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, 'mobilenetv3_confusion_matrix.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path

# 分析分类错误分布
def analyze_error_distribution(confusion_matrix):
    """分析分类错误分布"""
    num_classes = 9
    error_analysis = {}

    for true_label in range(num_classes):
        true_class = CLASS_NAMES[true_label]
        total_samples = np.sum(confusion_matrix[true_label, :])
        correct_samples = confusion_matrix[true_label, true_label]
        incorrect_samples = total_samples - correct_samples

        # 找出主要错误类别
        error_indices = np.argsort(confusion_matrix[true_label, :])[::-1]
        error_indices = [idx for idx in error_indices if idx != true_label]

        # 收集前3个主要错误
        top_errors = []
        for idx in error_indices[:3]:
            error_count = confusion_matrix[true_label, idx]
            if error_count > 0:
                top_errors.append({
                    'class': CLASS_NAMES[idx],
                    'count': error_count,
                    'percentage': (error_count / total_samples) * 100 if total_samples > 0 else 0
                })

        error_analysis[true_class] = {
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'incorrect_samples': incorrect_samples,
            'accuracy': (correct_samples / total_samples) * 100 if total_samples > 0 else 0,
            'top_errors': top_errors
        }

    return error_analysis

# 主函数
def main():
    """分析MobileNetV3模型的混淆矩阵"""
    print("开始分析MobileNetV3模型的混淆矩阵...")

    # 加载模型
    print("加载模型...")
    model = load_model()

    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建测试数据集和加载器
    print("加载测试数据集...")
    test_dataset = PathologicalDataset(TEST_LABELS_PATH, BASE_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 生成混淆矩阵
    print("生成混淆矩阵...")
    confusion_matrix = generate_confusion_matrix(model, test_loader)

    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    output_path = plot_confusion_matrix(confusion_matrix)
    print(f"混淆矩阵已保存至: {output_path}")

    # 分析错误分布
    print("分析分类错误分布...")
    error_analysis = analyze_error_distribution(confusion_matrix)

    # 打印错误分析结果
    print("\n分类错误分布分析:")
    print("-" * 100)
    for class_name, analysis in error_analysis.items():
        print(f"{class_name}:")
        print(f"  总样本数: {analysis['total_samples']}")
        print(f"  正确分类: {analysis['correct_samples']} ({analysis['accuracy']:.2f}%)")
        print(f"  错误分类: {analysis['incorrect_samples']}")

        if analysis['top_errors']:
            print("  主要错误类别:")
            for error in analysis['top_errors']:
                print(f"    - {error['class']}: {error['count']} ({error['percentage']:.2f}%)")
        print()

    print("分析完成！")

if __name__ == "__main__":
    main()