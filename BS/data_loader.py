#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器脚本

负责加载病理切片图像数据集，包括数据变换、自定义数据集类和构建DataLoader。
该脚本支持训练集、验证集和测试集的加载，并为训练集添加数据增强。
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from config import *

# 类别映射（字符串转数字）
CLASS_MAP = {
    "ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4,
    "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8
}

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 自定义数据集
class PathoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df_path = csv_path  # 保存CSV文件路径
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        cls_name = self.df.iloc[idx]["class"]
        cls_idx = CLASS_MAP[cls_name]

        # 构建绝对路径（基于CSV文件所在目录）
        import os
        csv_dir = os.path.dirname(self.df_path)
        img_path = os.path.join(csv_dir, img_path)

        # 加载.tif图片（转为RGB）
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(cls_idx, dtype=torch.long)


# 构建DataLoader
def get_dataloaders():
    train_dataset = PathoDataset(TRAIN_CSV, train_transform)
    val_dataset = PathoDataset(VAL_CSV, val_test_transform)
    test_dataset = PathoDataset(TEST_CSV, val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader