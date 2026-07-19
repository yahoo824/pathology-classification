#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析三类模型的训练损失随训练轮数的变化趋势
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置
LOG_DIR = r"D:\python_project\BS\model_checkpoints\new_models"
OUTPUT_DIR = r"D:\python_project\BS\model_checkpoints\analysis"

# 模型名称映射
MODEL_NAMES = {
    "resnet50": "ResNet50",
    "mobilenetv3": "MobileNetV3",
    "cnn_transformer": "CNN+Transformer"
}

# 颜色配置
COLORS = {
    "resnet50": {"train": "#1f77b4", "val": "#7570b3"},
    "mobilenetv3": {"train": "#2ca02c", "val": "#98df8a"},
    "cnn_transformer": {"train": "#ff7f0e", "val": "#ffbb78"}
}

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取日志文件
def read_log_file(model_name):
    """读取指定模型的训练日志"""
    log_path = os.path.join(LOG_DIR, f"{model_name}_metrics_log.csv")
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    else:
        print(f"警告：未找到{model_name}的日志文件")
        return None

# 分析损失趋势
def analyze_loss_trends():
    """分析三类模型的损失趋势"""
    # 读取所有模型的日志
    logs = {}
    for model_key in MODEL_NAMES:
        logs[model_key] = read_log_file(model_key)
    
    # 创建趋势图
    plt.figure(figsize=(12, 8))
    
    # 绘制每个模型的损失曲线
    for model_key, model_name in MODEL_NAMES.items():
        log_df = logs[model_key]
        if log_df is not None:
            # 提取训练轮数和损失
            epochs = log_df["Epoch"]
            train_loss = log_df["Train_Loss"]
            val_loss = log_df["Val_Loss"]
            
            # 绘制训练集损失
            plt.plot(epochs, train_loss, 
                     color=COLORS[model_key]["train"], 
                     linestyle="-", 
                     linewidth=2, 
                     marker="o", 
                     label=f"{model_name} 训练集")
            
            # 绘制验证集损失
            plt.plot(epochs, val_loss, 
                     color=COLORS[model_key]["val"], 
                     linestyle="--", 
                     linewidth=2, 
                     marker="s", 
                     label=f"{model_name} 验证集")
    
    # 设置图表属性
    plt.title("三类模型训练集/验证集损失趋势", fontsize=20, fontweight='bold')
    plt.xlabel("训练轮数 (Epoch)", fontsize=16)
    plt.ylabel("损失值", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=14, loc="upper right")
    plt.tight_layout()
    
    # 设置刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # 保存图表
    output_path = os.path.join(OUTPUT_DIR, "loss_trends.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 分析趋势
    print("三类模型损失趋势分析：")
    print("-" * 80)
    
    for model_key, model_name in MODEL_NAMES.items():
        log_df = logs[model_key]
        if log_df is not None:
            # 计算损失变化
            start_train_loss = log_df["Train_Loss"].iloc[0]
            end_train_loss = log_df["Train_Loss"].iloc[-1]
            start_val_loss = log_df["Val_Loss"].iloc[0]
            end_val_loss = log_df["Val_Loss"].iloc[-1]
            
            # 计算降低率
            train_reduction = (start_train_loss - end_train_loss) / start_train_loss * 100
            val_reduction = (start_val_loss - end_val_loss) / start_val_loss * 100
            
            print(f"{model_name}:")
            print(f"  训练集损失: {start_train_loss:.4f} → {end_train_loss:.4f} (降低 {train_reduction:.2f}%)")
            print(f"  验证集损失: {start_val_loss:.4f} → {end_val_loss:.4f} (降低 {val_reduction:.2f}%)")
            print(f"  最终训练集损失: {end_train_loss:.4f}")
            print(f"  最终验证集损失: {end_val_loss:.4f}")
            print(f"  验证集 vs 训练集: {'低于' if end_val_loss < end_train_loss else '高于' if end_val_loss > end_train_loss else '等于'} {abs(end_val_loss - end_train_loss):.4f}")
            print()
    
    print(f"趋势图已保存至: {output_path}")

if __name__ == "__main__":
    analyze_loss_trends()
