#!/usr/bin/env python3
"""
数据集分析脚本

功能：分析原始数据集的统计信息
输出：训练集、验证集、测试集的图像数量
      每个类别的分布情况
      总数据集统计信息
"""

import os

# ===================== 配置项 =====================
# 原始数据集路径
DATA_ROOT = "D:/processed_NCT-CRC-HE-100K"
# 类别列表
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
# 分割集
SPLITS = ["train", "val", "test"]

# ===================== 核心函数 =====================
def analyze_dataset():
    """分析数据集统计信息"""
    print("=" * 70)
    print("原始数据集统计分析")
    print("=" * 70)
    
    total_images = 0
    class_counts = {cls: 0 for cls in CLASSES}
    split_counts = {split: 0 for split in SPLITS}
    
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
    for cls, count in class_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {cls}: {count}张 ({percentage:.2f}%)")
    
    print("=" * 70)

# ===================== 执行 =====================
if __name__ == "__main__":
    analyze_dataset()
