#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集清理脚本

检测并删除数据集中的无效图片（如全黑图片），并生成清理后的标签文件。
"""

import os
import cv2
import pandas as pd
import numpy as np


def is_black_image(image_path, threshold=10):
    """检测图片是否为全黑"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True  # 无法读取的图片视为无效
        mean_value = np.mean(img)
        return mean_value < threshold
    except Exception:
        return True  # 处理异常，视为无效图片


def process_dataset_subset(data_dir, subset_name, labels_file, output_dir):
    """处理单个数据集子集"""
    # 读取标签文件
    df = pd.read_csv(labels_file)

    # 记录有效图片
    valid_rows = []
    deleted_count = 0

    # 遍历所有图片
    for idx, row in df.iterrows():
        # 构建图片路径
        if 'image_path' in row:
            # 如果标签文件中包含完整路径
            img_path = row['image_path']
        else:
            # 如果标签文件中只包含文件名
            img_path = os.path.join(data_dir, subset_name, row['image'])

        # 确保路径存在
        if not os.path.isabs(img_path):
            img_path = os.path.join(data_dir, img_path)

        if os.path.exists(img_path):
            if not is_black_image(img_path):
                valid_rows.append(row)
            else:
                # 删除无效图片
                try:
                    os.remove(img_path)
                    deleted_count += 1
                    print(f"已删除无效图片: {img_path}")
                except Exception as e:
                    print(f"删除图片失败: {img_path}, 错误: {e}")
        else:
            # 不存在的图片视为无效
            deleted_count += 1

    # 保存清理后的标签文件
    clean_df = pd.DataFrame(valid_rows)
    clean_labels_file = os.path.join(output_dir, f"clean_{subset_name}_labels.csv")
    clean_df.to_csv(clean_labels_file, index=False)

    # 记录统计信息
    stats = {
        'subset': subset_name,
        'original_count': len(df),
        'valid_count': len(valid_rows),
        'deleted_count': deleted_count
    }

    print(f"处理完成 {subset_name}:")
    print(f"  原始图片数: {stats['original_count']}")
    print(f"  有效图片数: {stats['valid_count']}")
    print(f"  删除图片数: {stats['deleted_count']}")
    print(f"  清理后标签文件: {clean_labels_file}")

    return stats


def main():
    # 配置参数
    DATA_DIR = r"E:\Dataset"  # 数据集根目录
    OUTPUT_DIR = r"E:\Dataset\cleaned"  # 输出目录
    SUBSETS = ['train', 'val', 'test']  # 数据集子集

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 处理每个子集
    all_stats = []
    for subset in SUBSETS:
        labels_file = os.path.join(DATA_DIR, f"{subset}_labels.csv")
        if os.path.exists(labels_file):
            stats = process_dataset_subset(DATA_DIR, subset, labels_file, OUTPUT_DIR)
            all_stats.append(stats)
        else:
            print(f"标签文件不存在: {labels_file}")

    # 生成统计报告
    stats_df = pd.DataFrame(all_stats)
    stats_file = os.path.join(OUTPUT_DIR, "cleaning_statistics.csv")
    stats_df.to_csv(stats_file, index=False)

    print("\n清理统计报告:")
    print(stats_df)
    print(f"\n统计报告保存到: {stats_file}")


if __name__ == "__main__":
    main()