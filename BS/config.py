#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件

包含数据集路径、模型超参数和训练配置等设置，为整个项目提供统一的配置管理。
"""

import os
import torch

# 数据集路径（新预处理的数据集）
TRAIN_CSV = r"E:\Dataset\train_labels.csv"
VAL_CSV = r"E:\Dataset\val_labels.csv"
TEST_CSV = r"E:\Dataset\test_labels.csv"
SAVE_DIR = r"D:\python_project\BS\model_checkpoints\new_models"  # 新模型保存路径

# 超参数
IMAGE_SIZE = (224, 224)  # 输入图片尺寸
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 9  # 9个类别（ADI/BACK/DEB/LYM/MUC/MUS/NORM/STR/TUM）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)