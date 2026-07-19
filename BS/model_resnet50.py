
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50模型训练与评估脚本

使用ResNet50模型进行病理切片图像分类，包括模型定义、训练、验证、测试和结果可视化。
该脚本包含正则化、学习率调度和早停机制，以提高模型性能和稳定性。
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

warnings.filterwarnings("ignore")

from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from config import *
from data_loader import get_dataloaders

# ===================== 全局配置（可视化保存路径） =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False
VIS_SAVE_PATH = os.path.join(SAVE_DIR, "resnet50_vis")
LOG_SAVE_PATH = os.path.join(SAVE_DIR, "resnet50_metrics_log.csv")
os.makedirs(VIS_SAVE_PATH, exist_ok=True)


# ===================== 1. 指标计算核心函数 =====================
def calculate_metrics(y_true, y_pred, y_score, num_classes=NUM_CLASSES):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    return {
        "Accuracy": round(acc, 4),
        "macro-F1": round(f1, 4),
        "AUC-ROC": round(auc, 4)
    }


# ===================== 2. ResNet50模型定义【修复pretrained警告】 =====================
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.5):
        super().__init__()
        # ✅ 修复：替换废弃的 pretrained=True → 使用官方权重枚举类（torchvision 0.13+ 规范）
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.resnet50.fc.in_features
        # 添加Dropout层进行正则化
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet50(x)


# ===================== 3. 训练+验证函数 =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience=5):
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    log_data = {
        "Epoch": [], "Train_Loss": [], "Val_Loss": [],
        "Train_Acc": [], "Val_Acc": [],
        "Train_F1": [], "Val_F1": [],
        "Train_AUC": [], "Val_AUC": [],
        "Learning_Rate": []
    }
    counter = 0  # 早停计数器

    for epoch in range(epochs):
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_true, train_pred, train_score = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train, LR: {current_lr:.6f})")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            pred = torch.argmax(outputs, dim=1)
            score = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            train_true.extend(labels.cpu().numpy())
            train_pred.extend(pred.cpu().numpy())
            train_score.extend(score)

        train_metrics = calculate_metrics(train_true, train_pred, train_score)
        train_loss_avg = round(train_loss / len(train_true), 4)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_true, val_pred, val_score = [], [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} (Val)")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                pred = torch.argmax(outputs, dim=1)
                score = torch.softmax(outputs, dim=1).detach().cpu().numpy()

                val_true.extend(labels.cpu().numpy())
                val_pred.extend(pred.cpu().numpy())
                val_score.extend(score)

        val_metrics = calculate_metrics(val_true, val_pred, val_score)
        val_loss_avg = round(val_loss / len(val_true), 4)

        # 更新学习率
        scheduler.step(val_loss_avg)

        # 记录日志
        log_data["Epoch"].append(epoch + 1)
        log_data["Train_Loss"].append(train_loss_avg)
        log_data["Val_Loss"].append(val_loss_avg)
        log_data["Train_Acc"].append(train_metrics["Accuracy"])
        log_data["Val_Acc"].append(val_metrics["Accuracy"])
        log_data["Train_F1"].append(train_metrics["macro-F1"])
        log_data["Val_F1"].append(val_metrics["macro-F1"])
        log_data["Train_AUC"].append(train_metrics["AUC-ROC"])
        log_data["Val_AUC"].append(val_metrics["AUC-ROC"])
        log_data["Learning_Rate"].append(current_lr)

        # 打印指标
        print(f"\n===== Epoch {epoch + 1} 指标汇总 =====")
        print(
            f"训练集 | Loss: {train_loss_avg:.4f} | Acc: {train_metrics['Accuracy']:.4f} | F1: {train_metrics['macro-F1']:.4f} | AUC: {train_metrics['AUC-ROC']:.4f}")
        print(
            f"验证集 | Loss: {val_loss_avg:.4f} | Acc: {val_metrics['Accuracy']:.4f} | F1: {val_metrics['macro-F1']:.4f} | AUC: {val_metrics['AUC-ROC']:.4f}")
        print(f"学习率: {current_lr:.6f}")

        # 早停检查
        if val_metrics["Accuracy"] > best_val_acc:
            best_val_acc = val_metrics["Accuracy"]
            best_val_f1 = val_metrics["macro-F1"]
            best_val_auc = val_metrics["AUC-ROC"]
            counter = 0  # 重置计数器
            
            # 添加时间戳避免覆盖旧模型
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"resnet50_best_{timestamp}.pth"
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_val_acc,
                "best_f1": best_val_f1,
                "best_auc": best_val_auc
            }, os.path.join(SAVE_DIR, model_filename))
            print(f"✅ 更新最优模型 | 文件名: {model_filename} | 最优Acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f} | AUC: {best_val_auc:.4f}\n")
        else:
            counter += 1
            print(f"⚠️  早停计数器: {counter}/{patience}\n")
            if counter >= patience:
                print(f"\n🛑 早停触发：验证集性能连续{patience}轮无提升")
                break

    return pd.DataFrame(log_data)


# ===================== 4. 结果可视化函数 =====================
def plot_metrics(log_df):
    epochs = log_df["Epoch"].values
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("ResNet50 病理切片9分类 - 训练全过程指标可视化", fontsize=16, fontweight="bold")

    # 损失曲线
    ax1.plot(epochs, log_df["Train_Loss"], "r-", linewidth=2, label="训练集损失")
    ax1.plot(epochs, log_df["Val_Loss"], "b-", linewidth=2, label="验证集损失")
    ax1.set_title("训练损失 vs 验证损失", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # 准确率曲线
    ax2.plot(epochs, log_df["Train_Acc"], "r-", linewidth=2, label="训练集准确率")
    ax2.plot(epochs, log_df["Val_Acc"], "b-", linewidth=2, label="验证集准确率")
    ax2.set_title("训练准确率 vs 验证准确率", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # F1曲线
    ax3.plot(epochs, log_df["Train_F1"], "r-", linewidth=2, label="训练集macro-F1")
    ax3.plot(epochs, log_df["Val_F1"], "b-", linewidth=2, label="验证集macro-F1")
    ax3.set_title("训练F1 vs 验证F1（类别平衡）", fontsize=12)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("macro-F1")
    ax3.legend(loc="best")
    ax3.grid(True, linestyle="--", alpha=0.7)

    # 学习率曲线
    ax4.plot(epochs, log_df["Learning_Rate"], "g-", linewidth=2, label="学习率")
    ax4.set_title("学习率变化曲线", fontsize=12)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Learning Rate")
    ax4.set_yscale("log")  # 使用对数尺度
    ax4.legend(loc="best")
    ax4.grid(True, linestyle="--", alpha=0.7)

    save_path = os.path.join(VIS_SAVE_PATH, "train_metrics_curve.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 训练可视化曲线已保存：{save_path}")


# ===================== 5. 主函数【修复torch.load加载失败】 =====================
if __name__ == "__main__":
    # 加载数据
    train_loader, val_loader, test_loader = get_dataloaders()
    print(
        f"数据加载完成 | 训练集Batch数：{len(train_loader)} | 验证集Batch数：{len(val_loader)} | 测试集Batch数：{len(test_loader)}")

    # 初始化模型
    model = ResNet50Model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # 添加权重衰减进行正则化
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    print(f"模型初始化完成 | 使用设备：{DEVICE}")

    # 训练模型
    metrics_log_df = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)

    # 保存日志+绘制可视化
    metrics_log_df.to_csv(LOG_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 训练指标日志已保存：{LOG_SAVE_PATH}")
    plot_metrics(metrics_log_df)

    # ===================== 测试集评估【✅ 核心修复：移除weights_only=True，解决加载失败】 =====================
    # 查找最新保存的模型文件
    model_files = glob.glob(os.path.join(SAVE_DIR, "resnet50_best_*.pth"))
    if model_files:
        model_path = max(model_files, key=os.path.getmtime)
        print(f"\n===== 加载最新模型文件：{os.path.basename(model_path)} =====")
    else:
        model_path = os.path.join(SAVE_DIR, "resnet50_best.pth")
        print(f"\n===== 加载默认模型文件：{os.path.basename(model_path)} =====")
    
    # ✅ 修复：删除 weights_only=True → 兼容numpy标量数据，正常加载模型（你的环境安全，放心使用）
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"===== 加载最优模型（Epoch {checkpoint['epoch']}）开始测试 =====")

    # 测试集推理
    test_true, test_pred, test_score = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="测试集推理中"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            pred = torch.argmax(outputs, dim=1)
            score = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            test_true.extend(labels.cpu().numpy())
            test_pred.extend(pred.cpu().numpy())
            test_score.extend(score)

    # 打印最终结果
    test_metrics = calculate_metrics(test_true, test_pred, test_score)
    print("\n" + "=" * 70)
    print("🎯 ResNet50 病理切片9分类 - 最终测试集性能指标（核心结论）")
    print("=" * 70)
    print(f"✅ 整体准确率 (Accuracy) ：{test_metrics['Accuracy']:.4f} ({test_metrics['Accuracy'] * 100:.2f}%)")
    print(f"✅ 类别平衡指标 (macro-F1)：{test_metrics['macro-F1']:.4f}")
    print(f"✅ 阈值鲁棒性 (AUC-ROC)   ：{test_metrics['AUC-ROC']:.4f}")
    print(f"✅ 最优验证集准确率       ：{checkpoint['best_acc']:.4f} ({checkpoint['best_acc'] * 100:.2f}%)")
    print("=" * 70)

    print(f"\n🎉 所有任务完成！结果文件汇总：")
    print(f"1. 最优模型权重：{model_path}")
    print(f"2. 训练指标日志：{LOG_SAVE_PATH}")
    print(f"3. 可视化曲线图：{os.path.join(VIS_SAVE_PATH, 'train_metrics_curve.png')}")