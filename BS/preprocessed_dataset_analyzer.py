#!/usr/bin/env python3
"""
预处理后数据集分析脚本

功能：
1. 统计预处理后数据集的9类病理组织样本量
2. 绘制柱状图展示训练集、验证集、测试集的样本分布
3. 计算各类组织的占比
4. 生成统计报告

输入：E:/Dataset（预处理后数据集）
输出：统计报告和可视化图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===================== 配置项 =====================
# 预处理后数据集路径
DATA_ROOT = "E:/Dataset"
# 类别列表
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
# 分割集
SPLITS = ["train", "val", "test"]
# 图表保存路径
CHART_SAVE_PATH = "E:/Dataset/distribution_chart.png"
# 统计报告保存路径
REPORT_SAVE_PATH = "E:/Dataset/dataset_statistics.csv"

# ===================== 核心函数 =====================
def analyze_preprocessed_dataset():
    """分析预处理后数据集"""
    print("=" * 70)
    print("预处理后数据集统计分析")
    print("=" * 70)
    
    # 初始化统计数据
    total_images = 0
    class_counts = {cls: 0 for cls in CLASSES}
    split_counts = {split: 0 for split in SPLITS}
    split_class_counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    
    # 分析每个分割集
    for split in SPLITS:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path):
            print(f"警告: {split_path} 不存在")
            continue
        
        split_total = 0
        print(f"\n{split}集:")
        
        # 分析每个类别
        for cls in CLASSES:
            cls_path = os.path.join(split_path, cls)
            if os.path.exists(cls_path):
                # 计算图像数量
                img_files = [f for f in os.listdir(cls_path) if f.lower().endswith((".tif", ".tiff"))]
                count = len(img_files)
                class_counts[cls] += count
                split_class_counts[split][cls] = count
                split_total += count
                total_images += count
                print(f"  {cls}: {count}张")
            else:
                print(f"  {cls}: 0张 (目录不存在)")
        
        split_counts[split] = split_total
        print(f"  总计: {split_total}张")
    
    # 输出总统计
    print(f"\n总数据集: {total_images}张图像")
    print("各分割集分布:")
    for split, count in split_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {split}: {count}张 ({percentage:.2f}%)")
    
    print("\n各类别分布:")
    class_percentages = {}
    for cls, count in class_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        class_percentages[cls] = percentage
        print(f"  {cls}: {count}张 ({percentage:.2f}%)")
    
    print("=" * 70)
    
    # 生成统计报告
    generate_statistics_report(class_counts, split_counts, split_class_counts, class_percentages, total_images)
    
    # 绘制分布图表
    plot_distribution_chart(split_class_counts, class_percentages)

def generate_statistics_report(class_counts, split_counts, split_class_counts, class_percentages, total_images):
    """生成统计报告"""
    # 创建数据框
    data = []
    for cls in CLASSES:
        row = {
            "Class": cls,
            "Total": class_counts[cls],
            "Percentage": class_percentages[cls],
            "Train": split_class_counts["train"][cls],
            "Val": split_class_counts["val"][cls],
            "Test": split_class_counts["test"][cls]
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 添加总计行
    total_row = {
        "Class": "Total",
        "Total": total_images,
        "Percentage": 100.0,
        "Train": split_counts["train"],
        "Val": split_counts["val"],
        "Test": split_counts["test"]
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # 保存为CSV
    df.to_csv(REPORT_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"\n统计报告已保存至: {REPORT_SAVE_PATH}")

def plot_distribution_chart(split_class_counts, class_percentages):
    """绘制分布图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    
    # 准备数据
    x = np.arange(len(CLASSES))
    width = 0.25
    
    # 分割集数据
    train_data = [split_class_counts["train"][cls] for cls in CLASSES]
    val_data = [split_class_counts["val"][cls] for cls in CLASSES]
    test_data = [split_class_counts["test"][cls] for cls in CLASSES]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个子图：各分割集样本分布
    ax1.bar(x - width, train_data, width, label='训练集', color='#1f77b4')
    ax1.bar(x, val_data, width, label='验证集', color='#ff7f0e')
    ax1.bar(x + width, test_data, width, label='测试集', color='#2ca02c')
    
    # 设置字体大小
    ax1.set_xlabel('病理组织类别', fontsize=20)
    ax1.set_ylabel('样本数量', fontsize=20)
    ax1.set_title('预处理后数据集各分割集样本分布', fontsize=22, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=18)
    for label in ax1.get_yticklabels():
        label.set_fontsize(18)
    ax1.legend(fontsize=18)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 第二个子图：各类别占比
    percentages = [class_percentages[cls] for cls in CLASSES]
    ax2.bar(x, percentages, width=0.6, color='#9467bd')
    
    ax2.set_xlabel('病理组织类别', fontsize=20)
    ax2.set_ylabel('占比 (%)', fontsize=20)
    ax2.set_title('各类别样本占比', fontsize=22, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=18)
    for label in ax2.get_yticklabels():
        label.set_fontsize(18)
    ax2.set_ylim(0, max(percentages) * 1.2)
    
    # 添加百分比标签
    for i, v in enumerate(percentages):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=18)
    
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(CHART_SAVE_PATH, bbox_inches='tight')
    plt.close()
    print(f"分布图表已保存至: {CHART_SAVE_PATH}")

# ===================== 执行 =====================
if __name__ == "__main__":
    analyze_preprocessed_dataset()
