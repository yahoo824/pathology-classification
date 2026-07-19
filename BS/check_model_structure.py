#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看模型权重文件的结构
"""

import torch

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"D:\python_project\BS\model_checkpoints\new_models"

def main():
    model_path = f"{MODEL_DIR}\\cnn_transformer_best_20260320_025434.pth"
    
    print(f"正在加载权重文件: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    state_dict = checkpoint["model_state_dict"]
    
    print("\n权重文件中的键列表:")
    print("="*80)
    
    keys = list(state_dict.keys())
    for i, key in enumerate(keys):
        param = state_dict[key]
        print(f"{i+1:3d}. {key:<50} shape: {tuple(param.shape)}")
    
    print("\n" + "="*80)
    print(f"总共有 {len(keys)} 个参数")

if __name__ == "__main__":
    main()
