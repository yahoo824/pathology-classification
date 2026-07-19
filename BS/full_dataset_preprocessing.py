#!/usr/bin/env python3
"""
完整病理切片数据集预处理脚本

功能：对原始数据集进行全流程预处理
处理步骤：
1. 数据清洗（移除损坏文件、过滤异常图像）
2. Macenko染色标准化（统一HE染色风格）
3. 尺寸归一化（224×224像素）
4. 病理专属数据增强（仅训练集）
5. 像素值标准化（均值-标准差归一化）

输入：D:/processed_NCT-CRC-HE-100K（原始数据集）
输出：E:/Dataset（预处理后数据集）
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random

# ===================== 配置项 =====================
# 原始数据集路径（包含train/val/test）
ORIGINAL_DATA_ROOT = "D:/processed_NCT-CRC-HE-100K"
# 预处理后保存路径
PREPROCESSED_DATA_ROOT = "E:/Dataset"
# 图像尺寸
IMG_SIZE = (224, 224)
# 类别列表
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
# 数据增强参数
AUGMENTATION_PROB = 0.5

# ===================== 病理图像预处理工具类 =====================
class CRCImageProcessor:
    """病理图像预处理工具类"""

    @staticmethod
    def macenko_stain_normalization(img, 
                                    target_means=[0.733, 0.561, 0.631], 
                                    target_stds=[0.163, 0.214, 0.147]):
        """Macenko染色标准化算法：统一HE染色风格"""
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img_float = img.astype(np.float32) / 255.0

        od = -np.log10(img_float + 1e-8)
        od = np.clip(od, 0, 10)

        od_reshaped = od.reshape(-1, 3)
        valid_mask = np.any(od_reshaped > 0.1, axis=1)
        od_valid = od_reshaped[valid_mask]

        if len(od_valid) < 20:
            return np.clip(img_float * 255, 0, 255).astype(np.uint8)

        cov_matrix = np.cov(od_valid.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvals)[::-1]
        eigenvecs_sorted = eigenvecs[:, sorted_idx]
        stain_vector = eigenvecs_sorted[:, :2]

        if stain_vector[0, 0] > 0:
            stain_vector[:, 0] *= -1
        if stain_vector[1, 1] > 0:
            stain_vector[:, 1] *= -1

        od_proj = od_valid @ stain_vector
        min_conc = np.percentile(od_proj, 1, axis=0)
        max_conc = np.percentile(od_proj, 99, axis=0)
        conc_diff = max_conc - min_conc
        conc_diff[conc_diff < 1e-6] = 1e-6

        conc_norm = (od_proj - min_conc) / conc_diff
        conc_norm = np.clip(conc_norm, 0, 1)
        conc_norm = conc_norm * np.array(target_stds[:2]) + np.array(target_means[:2])

        od_norm = conc_norm @ stain_vector.T
        od_norm = np.clip(od_norm, 0, 5)

        img_norm = 10 ** (-od_norm)
        img_norm = np.clip(img_norm, 0, 1)

        img_norm_reshaped = np.ones_like(od)
        img_norm_reshaped[valid_mask.reshape(od.shape[:2])] = img_norm

        return np.clip(img_norm_reshaped * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def extract_tissue_region(img):
        """提取有效组织区域，去除空白背景"""
        if len(img.shape) != 3:
            return img

        # 转换为HSV色彩空间（手动实现）
        img_float = img.astype(np.float32) / 255.0
        r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        delta = max_rgb - min_rgb
        
        # 计算饱和度通道
        s_channel = np.zeros_like(max_rgb)
        mask = max_rgb > 0
        s_channel[mask] = delta[mask] / max_rgb[mask]
        s_channel = (s_channel * 255).astype(np.uint8)
        
        # 阈值处理
        tissue_mask = (s_channel > 15).astype(np.uint8) * 255
        
        # 简单的形态学操作
        from scipy.ndimage import binary_closing
        tissue_mask = binary_closing(tissue_mask, structure=np.ones((5, 5))).astype(np.uint8) * 255
        
        # 应用掩码
        tissue_img = img.copy()
        tissue_img[tissue_mask == 0] = 0
        
        return tissue_img

    @staticmethod
    def normalize_pixels(img):
        """像素值标准化：均值-标准差归一化"""
        img_float = img.astype(np.float32) / 255.0
        mean = np.mean(img_float, axis=(0, 1))
        std = np.std(img_float, axis=(0, 1))
        std = np.maximum(std, 1e-8)  # 避免除零
        img_normalized = (img_float - mean) / std
        return img_normalized

# ===================== 数据增强工具 =====================
class CRCDataAugmenter:
    """病理图像数据增强工具"""

    @staticmethod
    def random_flip(img):
        """随机翻转"""
        if random.random() > 0.5:
            img = np.fliplr(img)  # 水平翻转
        if random.random() > 0.5:
            img = np.flipud(img)  # 垂直翻转
        return img

    @staticmethod
    def random_rotation(img, max_angle=15):
        """随机旋转"""
        angle = random.uniform(-max_angle, max_angle)
        img_pil = Image.fromarray(img)
        img_rotated = img_pil.rotate(angle, fillcolor=(0, 0, 0))
        return np.array(img_rotated)

    @staticmethod
    def random_brightness_contrast(img, max_delta=0.15):
        """随机亮度和对比度调整"""
        img_float = img.astype(np.float32) / 255.0
        brightness = 1.0 + random.uniform(-max_delta, max_delta)
        contrast = 1.0 + random.uniform(-max_delta, max_delta)
        img_augmented = (img_float * contrast + brightness)
        img_augmented = np.clip(img_augmented, 0, 1) * 255
        return img_augmented.astype(np.uint8)

    @staticmethod
    def apply_augmentation(img):
        """应用数据增强"""
        img = CRCDataAugmenter.random_flip(img)
        img = CRCDataAugmenter.random_rotation(img)
        img = CRCDataAugmenter.random_brightness_contrast(img)
        return img

# ===================== 数据清洗工具 =====================
class CRCDataCleaner:
    """数据清洗工具"""

    @staticmethod
    def is_valid_image(img_path):
        """检查图像是否有效"""
        try:
            with Image.open(img_path) as img:
                img.verify()  # 验证图像完整性
            # 再次打开确认
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            # 检查图像尺寸
            if img_array.shape[0] < 64 or img_array.shape[1] < 64:
                return False
            # 检查图像内容（避免全黑或全白图像）
            mean_brightness = np.mean(img_array)
            if mean_brightness < 10 or mean_brightness > 245:
                return False
            return True
        except Exception:
            return False

# ===================== 核心函数 =====================
def create_output_directory():
    """创建输出目录结构"""
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            output_dir = os.path.join(PREPROCESSED_DATA_ROOT, split, cls)
            os.makedirs(output_dir, exist_ok=True)

def preprocess_single_image(img_path, output_path, split_type="train"):
    """预处理单张图像"""
    try:
        # 1. 数据清洗：检查图像有效性
        if not CRCDataCleaner.is_valid_image(img_path):
            print(f"跳过无效图像: {img_path}")
            return False
        
        # 2. 读取图像
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        
        # 3. Macenko染色标准化
        processor = CRCImageProcessor()
        img = processor.macenko_stain_normalization(img)
        
        # 4. 组织区域提取
        img = processor.extract_tissue_region(img)
        
        # 5. 数据增强（仅训练集）
        if split_type == "train" and random.random() < AUGMENTATION_PROB:
            img = CRCDataAugmenter.apply_augmentation(img)
        
        # 6. 尺寸归一化
        img = Image.fromarray(img)
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img = np.array(img)
        
        # 7. 像素值标准化（保存时会转换回0-255范围）
        img_normalized = processor.normalize_pixels(img)
        # 将标准化后的值映射回0-255范围以便保存
        img_normalized = ((img_normalized - img_normalized.min()) / 
                         (img_normalized.max() - img_normalized.min() + 1e-8) * 255)
        img_normalized = img_normalized.astype(np.uint8)
        
        # 8. 保存预处理后的图像
        img_final = Image.fromarray(img_normalized)
        img_final.save(output_path)
        return True
    except Exception as e:
        print(f"处理图像失败 {img_path}: {e}")
        return False

def preprocess_dataset():
    """预处理整个数据集"""
    print("=" * 70)
    print("开始预处理数据集...")
    print(f"原始数据路径: {ORIGINAL_DATA_ROOT}")
    print(f"预处理后保存路径: {PREPROCESSED_DATA_ROOT}")
    print(f"目标尺寸: {IMG_SIZE}")
    print("=" * 70)
    
    # 创建输出目录
    create_output_directory()
    
    # 统计信息
    total_processed = 0
    total_failed = 0
    total_skipped = 0
    
    # 遍历所有分割和类别
    for split in ["train", "val", "test"]:
        print(f"\n处理 {split} 集...")
        
        for cls in CLASSES:
            input_dir = os.path.join(ORIGINAL_DATA_ROOT, split, cls)
            output_dir = os.path.join(PREPROCESSED_DATA_ROOT, split, cls)
            
            if not os.path.exists(input_dir):
                print(f"警告: {input_dir} 不存在，跳过")
                continue
            
            # 获取所有图像文件
            img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))]
            print(f"  类别 {cls}: {len(img_files)} 张图像")
            
            # 处理每张图像
            for img_file in tqdm(img_files, desc=f"  处理 {cls}"):
                input_path = os.path.join(input_dir, img_file)
                output_path = os.path.join(output_dir, img_file)
                
                if preprocess_single_image(input_path, output_path, split):
                    total_processed += 1
                else:
                    total_failed += 1
    
    # 输出统计结果
    print("\n" + "=" * 70)
    print("预处理完成！")
    print(f"总处理图像数: {total_processed}")
    print(f"失败图像数: {total_failed}")
    print(f"成功率: {total_processed / (total_processed + total_failed) * 100:.2f}%")
    print(f"预处理后数据路径: {PREPROCESSED_DATA_ROOT}")
    print("=" * 70)

# ===================== 数据集统计工具 =====================
def analyze_dataset():
    """分析数据集统计信息"""
    print("\n" + "=" * 70)
    print("数据集统计分析")
    print("=" * 70)
    
    total_images = 0
    class_counts = {cls: 0 for cls in CLASSES}
    
    for split in ["train", "val", "test"]:
        split_count = 0
        print(f"\n{split}集:")
        
        for cls in CLASSES:
            input_dir = os.path.join(ORIGINAL_DATA_ROOT, split, cls)
            if os.path.exists(input_dir):
                count = len([f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))])
                class_counts[cls] += count
                split_count += count
                total_images += count
                print(f"  {cls}: {count}张")
            else:
                print(f"  {cls}: 0张 (目录不存在)")
        
        print(f"  总计: {split_count}张")
    
    print(f"\n总数据集: {total_images}张图像")
    print("各类别分布:")
    for cls, count in class_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {cls}: {count}张 ({percentage:.2f}%)")
    print("=" * 70)

# ===================== 执行 =====================
if __name__ == "__main__":
    # 分析数据集
    analyze_dataset()
    # 预处理数据集
    preprocess_dataset()
