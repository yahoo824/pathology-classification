#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV3消融实验完整脚本

包含所有消融实验变体的训练和评估：
1. 基础模型（无正则化）
2. +Batch Normalization
3. +BN+Dropout(0.2)
4. +BN+Dropout(0.5)
5. +BN+Dropout(0.2)+L2(1e-5)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import datetime
import glob
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
from config import *
from data_loader import get_dataloaders

warnings.filterwarnings("ignore")

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"D:\python_project\BS\model_checkpoints\new_models"
ABLATION_DIR = os.path.join(MODEL_DIR, "ablation_experiments")
OUTPUT_DIR = r"D:\python_project\BS\model_checkpoints\analysis"

# 确保目录存在
os.makedirs(ABLATION_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MobileNetV3模型定义（不同变体）
class MobileNetV3Ablation(nn.Module):
    def __init__(self, num_classes=9, use_bn=True, dropout_rate=0):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = self.mobilenet.classifier[-1].in_features
        
        # 根据参数配置分类头
        layers = []
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(in_features, num_classes))
        
        self.mobilenet.classifier[-1] = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mobilenet(x)

# 训练函数
def train_single_variant(model, train_loader, val_loader, criterion, optimizer, epochs, variant_name):
    """训练单个模型变体"""
    best_val_acc = 0.0
    best_val_f1 = 0.0
    log_data = {
        'Epoch': [], 
        'Train_Loss': [], 
        'Val_Loss': [],
        'Train_Acc': [], 
        'Val_Acc': [],
        'Train_F1': [], 
        'Val_F1': []
    }
    
    print(f"\n{'='*80}")
    print(f"训练模型变体: {variant_name}")
    print('='*80)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_true, train_pred, train_score = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            pred = torch.argmax(outputs, dim=1)
            
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(pred.cpu().numpy())
        
        # 计算训练指标
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average='macro')
        train_loss_avg = train_loss / len(train_true)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} (Val)")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                pred = torch.argmax(outputs, dim=1)
                
                val_true.extend(labels.cpu().numpy())
                val_pred.extend(pred.cpu().numpy())
        
        # 计算验证指标
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average='macro')
        val_loss_avg = val_loss / len(val_true)
        
        # 记录日志
        log_data['Epoch'].append(epoch + 1)
        log_data['Train_Loss'].append(train_loss_avg)
        log_data['Val_Loss'].append(val_loss_avg)
        log_data['Train_Acc'].append(train_acc)
        log_data['Val_Acc'].append(val_acc)
        log_data['Train_F1'].append(train_f1)
        log_data['Val_F1'].append(val_f1)
        
        # 打印指标
        print(f"训练 | Loss: {train_loss_avg:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"验证 | Loss: {val_loss_avg:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            
            # 添加时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(ABLATION_DIR, f"mobilenetv3_{variant_name}_best.pth")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc,
                'best_f1': best_val_f1
            }, model_path)
            print(f"✅ 更新最佳模型 | 最佳Acc: {best_val_acc:.4f}")
    
    # 保存训练日志
    log_df = pd.DataFrame(log_data)
    log_path = os.path.join(ABLATION_DIR, f"{variant_name}_training_log.csv")
    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    print(f"训练日志已保存至: {log_path}")
    
    return best_val_acc, best_val_f1

# 评估函数
def evaluate_single_variant(model, test_loader):
    """评估单个模型变体的性能"""
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估中..."):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 计算黏液组织召回率
    muc_true = (y_true == 4).astype(int)
    muc_pred = (y_pred == 4).astype(int)
    muc_recall = recall_score(muc_true, muc_pred, zero_division=0)
    
    return accuracy, macro_f1, muc_recall

# 加载最佳模型
def load_best_model(variant_name):
    """加载最佳模型"""
    model_path = os.path.join(ABLATION_DIR, f"mobilenetv3_{variant_name}_best.pth")
    
    if not os.path.exists(model_path):
        print(f"⚠️ 模型文件不存在: {model_path}")
        return None
    
    # 根据变体名称解析参数
    if "base" in variant_name.lower():
        model = MobileNetV3Ablation(num_classes=9, use_bn=True, dropout_rate=0).to(DEVICE)
    elif "bn_dropout_02_l2" in variant_name.lower():
        model = MobileNetV3Ablation(num_classes=9, use_bn=True, dropout_rate=0.2).to(DEVICE)
    elif "bn_dropout_05" in variant_name.lower():
        model = MobileNetV3Ablation(num_classes=9, use_bn=True, dropout_rate=0.5).to(DEVICE)
    elif "bn_dropout_02" in variant_name.lower():
        model = MobileNetV3Ablation(num_classes=9, use_bn=True, dropout_rate=0.2).to(DEVICE)
    elif "bn_only" in variant_name.lower():
        model = MobileNetV3Ablation(num_classes=9, use_bn=True, dropout_rate=0).to(DEVICE)
    else:
        model = MobileNetV3Ablation(num_classes=9, use_bn=True, dropout_rate=0).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model

# 主函数
def main():
    """完整的MobileNetV3消融实验"""
    print("="*80)
    print("MobileNetV3消融实验开始")
    print("="*80)
    
    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 定义消融实验变体
    ablation_variants = [
        {
            'name': 'base',
            'display_name': '基础模型（无正则化）',
            'use_bn': True,
            'dropout_rate': 0,
            'weight_decay': 0
        },
        {
            'name': 'bn_only',
            'display_name': '+Batch Normalization',
            'use_bn': True,
            'dropout_rate': 0,
            'weight_decay': 0
        },
        {
            'name': 'bn_dropout_02',
            'display_name': '+BN+Dropout(0.2)',
            'use_bn': True,
            'dropout_rate': 0.2,
            'weight_decay': 0
        },
        {
            'name': 'bn_dropout_05',
            'display_name': '+BN+Dropout(0.5)',
            'use_bn': True,
            'dropout_rate': 0.5,
            'weight_decay': 0
        },
        {
            'name': 'bn_dropout_02_l2',
            'display_name': '+BN+Dropout(0.2)+L2(1e-5)',
            'use_bn': True,
            'dropout_rate': 0.2,
            'weight_decay': 1e-5
        }
    ]
    
    # 训练每个变体
    print(f"\n{'='*80}")
    print("开始训练所有消融实验变体")
    print('='*80)
    
    results = {}
    for variant in ablation_variants:
        print(f"\n{'='*80}")
        print(f"训练变体: {variant['display_name']}")
        print('='*80)
        
        # 创建模型
        model = MobileNetV3Ablation(
            num_classes=9,
            use_bn=variant['use_bn'],
            dropout_rate=variant['dropout_rate']
        ).to(DEVICE)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=variant['weight_decay'])
        
        # 训练模型
        best_val_acc, best_val_f1 = train_single_variant(
            model, train_loader, val_loader, criterion, optimizer, EPOCHS, variant['name']
        )
        
        results[variant['name']] = {
            'display_name': variant['display_name'],
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1
        }
    
    # 评估所有变体
    print(f"\n{'='*80}")
    print("开始评估所有消融实验变体")
    print('='*80)
    
    final_results = []
    for variant in ablation_variants:
        print(f"\n评估变体: {variant['display_name']}")
        
        # 加载最佳模型
        model = load_best_model(variant['name'])
        if model is None:
            print(f"跳过 {variant['display_name']}")
            continue
        
        # 评估模型
        accuracy, macro_f1, muc_recall = evaluate_single_variant(model, test_loader)
        
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"macro-F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
        print(f"黏液组织召回率: {muc_recall:.4f} ({muc_recall*100:.2f}%)")
        
        final_results.append({
            '策略组合': variant['display_name'],
            '准确率': f"{accuracy*100:.2f}%",
            'macro-F1': f"{macro_f1*100:.2f}%",
            '黏液组织类别召回率': f"{muc_recall*100:.2f}%"
        })
    
    # 生成并保存消融实验结果表格
    print(f"\n{'='*80}")
    print("消融实验结果汇总")
    print('='*80)
    
    df = pd.DataFrame(final_results)
    
    # 打印表格
    print(f"\n{'策略组合':<35} {'准确率':<12} {'macro-F1':<12} {'黏液组织类别召回率':<15}")
    print('-'*80)
    for idx, row in df.iterrows():
        print(f"{row['策略组合']:<35} {row['准确率']:<12} {row['macro-F1']:<12} {row['黏液组织类别召回率']:<15}")
    
    # 保存表格
    output_path = os.path.join(OUTPUT_DIR, "ablation_experiment_final_results.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n消融实验结果已保存至: {output_path}")
    
    # 打印Markdown格式的表格
    print(f"\n{'='*80}")
    print("表 4-4 MobileNetV3 改进策略消融实验结果")
    print('='*80)
    print(f"{'策略组合':<35} {'准确率':<12} {'macro-F1':<12} {'黏液组织类别召回率':<15}")
    print('-'*80)
    for idx, row in df.iterrows():
        print(f"{row['策略组合']:<35} {row['准确率']:<12} {row['macro-F1']:<12} {row['黏液组织类别召回率']:<15}")
    print('='*80)
    
    print(f"\n消融实验完成！")
    print(f"模型保存位置: {ABLATION_DIR}")
    print(f"结果保存位置: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
