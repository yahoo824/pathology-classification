#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN+Transformer混合模型训练与评估脚本

使用CNN+Transformer混合架构进行病理切片图像分类，包括模型定义、训练、验证、测试和结果可视化。
该脚本使用ResNet50作为CNN骨干网络，结合Transformer进行全局特征建模，提高模型的分类性能。
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

import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from config import *
from data_loader import get_dataloaders

# ===================== 全局配置 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
VIS_SAVE_PATH = os.path.join(SAVE_DIR, "cnn_transformer_vis")
LOG_SAVE_PATH = os.path.join(SAVE_DIR, "cnn_transformer_metrics_log.csv")
os.makedirs(VIS_SAVE_PATH, exist_ok=True)


# ===================== 指标计算函数 =====================
def calculate_metrics(y_true, y_pred, y_score, num_classes=NUM_CLASSES):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    return {
        "Accuracy": round(acc, 4),
        "macro-F1": round(f1, 4),
        "AUC-ROC": round(auc, 4)
    }


# ===================== CNN+Transformer核心模块【✅ 终极适配：2048通道+7×7尺寸】 =====================
class PatchEmbedding(nn.Module):
    """✅ 固定适配：输入2048通道 + 7×7尺寸，patch_size=2 → 输出3×3序列"""

    def __init__(self):
        super().__init__()
        # ✅ 核心：2048输入通道 + 2×2卷积核 → 完美匹配7×7输入，输出3×3特征
        self.proj = nn.Conv2d(2048, 512, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.proj(x)  # [B,2048,7,7] → [B,512,3,3] ✅ 尺寸完全匹配
        x = x.flatten(2)  # [B,512,9]
        x = x.transpose(1, 2)  # [B,9,512]
        return x


class TransformerEncoder(nn.Module):
    """✅ 轻量化编码器：适配GPU，提速训练"""

    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ===================== CNN+Transformer混合模型【✅ 完全适配你的环境，零改动】 =====================
class CNNTransformerModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # ✅ 固定适配：ResNet50完整主干，输出2048通道+7×7尺寸（匹配你的环境）
        self.cnn_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 移除最后的全局池化和分类层，保留特征提取部分
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])

        # ✅ 冻结大部分权重，仅训练后10层，提速+降显存
        for i, param in enumerate(self.cnn_backbone.parameters()):
            param.requires_grad = i > 140

        # ✅ 加载适配好的分块嵌入层
        self.patch_embed = PatchEmbedding()
        self.transformer_encoder = TransformerEncoder(512, 4)

        # ✅ 分类头：适配9分类任务
        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Step1: CNN提取特征 → [B,2048,7,7] ✅ 完全匹配你的环境
        cnn_feat = self.cnn_backbone(x)
        # Step2: 分块嵌入 → [B,9,512]
        patch_feat = self.patch_embed(cnn_feat)
        # Step3: Transformer全局特征建模
        trans_feat = self.transformer_encoder(patch_feat)
        # Step4: 全局池化+分类
        global_feat = trans_feat.mean(dim=1)
        out = self.fc(global_feat)
        return out


# ===================== 训练+验证+可视化+测试（复用成熟框架，和前两个模型完全一致） =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    log_data = {
        "Epoch": [], "Train_Loss": [], "Val_Loss": [],
        "Train_Acc": [], "Val_Acc": [],
        "Train_F1": [], "Val_F1": [],
        "Train_AUC": [], "Val_AUC": []
    }

    for epoch in range(epochs):
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
            score = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(pred.cpu().numpy())
            train_score.extend(score)

        train_metrics = calculate_metrics(train_true, train_pred, train_score)
        train_loss_avg = round(train_loss / len(train_true), 4)

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

        log_data["Epoch"].append(epoch + 1)
        log_data["Train_Loss"].append(train_loss_avg)
        log_data["Val_Loss"].append(val_loss_avg)
        log_data["Train_Acc"].append(train_metrics["Accuracy"])
        log_data["Val_Acc"].append(val_metrics["Accuracy"])
        log_data["Train_F1"].append(train_metrics["macro-F1"])
        log_data["Val_F1"].append(val_metrics["macro-F1"])
        log_data["Train_AUC"].append(train_metrics["AUC-ROC"])
        log_data["Val_AUC"].append(val_metrics["AUC-ROC"])

        print(f"\n===== Epoch {epoch + 1} 指标汇总 =====")
        print(
            f"训练集 | Loss: {train_loss_avg:.4f} | Acc: {train_metrics['Accuracy']:.4f} | F1: {train_metrics['macro-F1']:.4f} | AUC: {train_metrics['AUC-ROC']:.4f}")
        print(
            f"验证集 | Loss: {val_loss_avg:.4f} | Acc: {val_metrics['Accuracy']:.4f} | F1: {val_metrics['macro-F1']:.4f} | AUC: {val_metrics['AUC-ROC']:.4f}")

        if val_metrics["Accuracy"] > best_val_acc:
            best_val_acc = val_metrics["Accuracy"]
            best_val_f1 = val_metrics["macro-F1"]
            best_val_auc = val_metrics["AUC-ROC"]
            
            # 添加时间戳避免覆盖旧模型
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"cnn_transformer_best_{timestamp}.pth"
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_val_acc,
                "best_f1": best_val_f1,
                "best_auc": best_val_auc
            }, os.path.join(SAVE_DIR, model_filename))
            print(f"✅ 更新最优模型 | 文件名: {model_filename} | 最优Acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f} | AUC: {best_val_auc:.4f}\n")

    return pd.DataFrame(log_data)


def plot_metrics(log_df):
    epochs = log_df["Epoch"].values
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CNN+Transformer 病理切片9分类 - 训练全过程指标可视化", fontsize=16, fontweight="bold")
    ax1.plot(epochs, log_df["Train_Loss"], "r-", linewidth=2, label="训练集损失")
    ax1.plot(epochs, log_df["Val_Loss"], "b-", linewidth=2, label="验证集损失")
    ax1.set_title("训练损失 vs 验证损失", fontsize=12);
    ax1.set_xlabel("Epoch");
    ax1.set_ylabel("Loss");
    ax1.legend();
    ax1.grid(True, alpha=0.7)
    ax2.plot(epochs, log_df["Train_Acc"], "r-", linewidth=2, label="训练集准确率")
    ax2.plot(epochs, log_df["Val_Acc"], "b-", linewidth=2, label="验证集准确率")
    ax2.set_title("训练准确率 vs 验证准确率", fontsize=12);
    ax2.set_xlabel("Epoch");
    ax2.set_ylabel("Accuracy");
    ax2.legend();
    ax2.grid(True, alpha=0.7)
    ax3.plot(epochs, log_df["Train_F1"], "r-", linewidth=2, label="训练集macro-F1")
    ax3.plot(epochs, log_df["Val_F1"], "b-", linewidth=2, label="验证集macro-F1")
    ax3.set_title("训练F1 vs 验证F1", fontsize=12);
    ax3.set_xlabel("Epoch");
    ax3.set_ylabel("macro-F1");
    ax3.legend();
    ax3.grid(True, alpha=0.7)
    ax4.plot(epochs, log_df["Train_AUC"], "r-", linewidth=2, label="训练集AUC-ROC")
    ax4.plot(epochs, log_df["Val_AUC"], "b-", linewidth=2, label="验证集AUC-ROC")
    ax4.set_title("训练AUC vs 验证AUC", fontsize=12);
    ax4.set_xlabel("Epoch");
    ax4.set_ylabel("AUC-ROC");
    ax4.legend();
    ax4.grid(True, alpha=0.7)
    save_path = os.path.join(VIS_SAVE_PATH, "train_metrics_curve.png")
    plt.tight_layout();
    plt.savefig(save_path, dpi=300, bbox_inches="tight");
    plt.close()
    print(f"\n✅ 训练可视化曲线已保存：{save_path}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(
        f"✅ 数据加载完成 | 训练集Batch数：{len(train_loader)} | 验证集Batch数：{len(val_loader)} | 测试集Batch数：{len(test_loader)}")

    model = CNNTransformerModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    print(f"✅ CNN+Transformer模型初始化完成 | 使用设备：{DEVICE}")

    metrics_log_df = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    metrics_log_df.to_csv(LOG_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 训练指标日志已保存：{LOG_SAVE_PATH}")
    plot_metrics(metrics_log_df)

    # 查找最新保存的模型文件
    model_files = glob.glob(os.path.join(SAVE_DIR, "cnn_transformer_best_*.pth"))
    if model_files:
        model_path = max(model_files, key=os.path.getmtime)
        print(f"\n===== 加载最新模型文件：{os.path.basename(model_path)} =====")
    else:
        model_path = os.path.join(SAVE_DIR, "cnn_transformer_best.pth")
        print(f"\n===== 加载默认模型文件：{os.path.basename(model_path)} =====")
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"===== 加载最优模型（Epoch {checkpoint['epoch']}）开始测试 =====")

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

    test_metrics = calculate_metrics(test_true, test_pred, test_score)
    print("\n" + "=" * 70)
    print("🎯 CNN+Transformer 病理切片9分类 - 最终测试集性能指标")
    print("=" * 70)
    print(f"✅ 整体准确率 (Accuracy) ：{test_metrics['Accuracy']:.4f} ({test_metrics['Accuracy'] * 100:.2f}%)")
    print(f"✅ 类别平衡指标 (macro-F1)：{test_metrics['macro-F1']:.4f}")
    print(f"✅ 阈值鲁棒性 (AUC-ROC)   ：{test_metrics['AUC-ROC']:.4f}")
    print(f"✅ 最优验证集准确率       ：{checkpoint['best_acc']:.4f} ({checkpoint['best_acc'] * 100:.2f}%)")
    print("=" * 70)

    print(f"\n🎉 训练完成！结果文件汇总：")
    print(f"1. 最优模型权重：{model_path}")
    print(f"2. 训练指标日志：{LOG_SAVE_PATH}")
    print(f"3. 可视化曲线图：{os.path.join(VIS_SAVE_PATH, 'train_metrics_curve.png')}")