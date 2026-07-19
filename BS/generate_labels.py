#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成数据集标签文件

从目录结构自动生成训练集、验证集和测试集的标签文件，并统计数据集信息。
"""

import os
import pandas as pd


def generate_labels_from_directory(data_dir, output_dir):
    """从目录结构生成标签文件"""
    # 定义数据集子集
    subsets = ['train', 'val', 'test']
    # 定义类别映射
    class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个子集
    all_stats = []
    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"子集目录不存在: {subset_dir}")
            continue

        # 存储标签数据
        labels_data = []

        # 遍历每个类别目录
        for class_name in class_names:
            class_dir = os.path.join(subset_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"类别目录不存在: {class_dir}")
                continue

            # 遍历目录中的图片文件
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.tif', '.jpg', '.jpeg', '.png')):
                    # 构建相对路径
                    relative_path = os.path.join(subset, class_name, img_file)
                    # 添加到标签数据
                    labels_data.append({
                        'image_path': relative_path,
                        'class': class_name
                    })

        # 生成标签文件
        if labels_data:
            df = pd.DataFrame(labels_data)
            label_file = os.path.join(output_dir, f"{subset}_labels.csv")
            df.to_csv(label_file, index=False)

            # 记录统计信息
            stats = {
                'subset': subset,
                'total_images': len(df),
                'classes': len(df['class'].unique())
            }
            all_stats.append(stats)

            print(f"生成标签文件: {label_file}")
            print(f"  图片数量: {stats['total_images']}")
            print(f"  类别数量: {stats['classes']}")
        else:
            print(f"子集 {subset} 中没有找到图片")

    # 生成统计报告
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_file = os.path.join(output_dir, "dataset_statistics.csv")
        stats_df.to_csv(stats_file, index=False)

        print("\n数据集统计报告:")
        print(stats_df)
        print(f"\n统计报告保存到: {stats_file}")
    else:
        print("没有生成任何标签文件")


def main():
    # 配置参数
    DATA_DIR = r"E:\Dataset"  # 数据集根目录
    OUTPUT_DIR = r"E:\Dataset"  # 输出目录（与数据集根目录相同）

    # 执行标签生成
    generate_labels_from_directory(DATA_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()