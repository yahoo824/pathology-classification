#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集检查脚本

检查训练集、验证集和测试集的基本信息，包括图片数量和类别分布。
"""

import pandas as pd

# 检查训练集
print("=== 训练集信息 ===")
train_df = pd.read_csv(r"E:\Dataset\train_labels.csv")
print(f"训练集图片数: {len(train_df)}")
print("类别分布:")
print(train_df['class'].value_counts())

# 检查验证集
print("\n=== 验证集信息 ===")
val_df = pd.read_csv(r"E:\Dataset\val_labels.csv")
print(f"验证集图片数: {len(val_df)}")
print("类别分布:")
print(val_df['class'].value_counts())

# 检查测试集
print("\n=== 测试集信息 ===")
test_df = pd.read_csv(r"E:\Dataset\test_labels.csv")
print(f"测试集图片数: {len(test_df)}")
print("类别分布:")
print(test_df['class'].value_counts())

# 总数据量
print("\n=== 总数据量 ===")
total = len(train_df) + len(val_df) + len(test_df)
print(f"总图片数: {total}")
