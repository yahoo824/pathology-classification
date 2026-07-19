#!/usr/bin/env python3
"""
模型性能统计显著性检验脚本
使用McNemar's test比较三个模型在测试集上的性能差异
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model_resnet50 import ResNet50Model
from model_mobilenetv3 import MobileNetV3Model
from model_cnn_transformer import CNNTransformerModel

# 配置路径
DATA_DIR = r"E:\Dataset\test"
MODEL_DIR = r"D:\python_project\BS\model_checkpoints\new_models"
OUTPUT_DIR = r"D:\python_project\BS\model_checkpoints\analysis"

def find_model_file(dir_path, model_prefix):
    """查找最新的模型文件"""
    # 先查找不带时间戳的文件
    simple_file = os.path.join(dir_path, f"{model_prefix}_best.pth")
    if os.path.exists(simple_file):
        return simple_file
    
    # 查找带时间戳的文件
    files = []
    for f in os.listdir(dir_path):
        if f.startswith(model_prefix) and f.endswith('.pth'):
            files.append(os.path.join(dir_path, f))
    
    if files:
        # 返回最新的文件（按修改时间排序）
        return sorted(files, key=os.path.getmtime, reverse=True)[0]
    return None

# 类别名称
CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
CLASS_NAMES_CN = ['脂肪组织', '背景组织', '碎片组织', '淋巴组织', '黏液组织', '肌肉组织', '正常黏膜', '基质组织', '肿瘤上皮']

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    """自定义批处理函数"""
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

class TestDataset(torch.utils.data.Dataset):
    """测试数据集"""
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # 遍历所有类别
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.tif'):
                    self.samples.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

def load_models():
    """加载三个模型"""
    print("正在加载模型...")
    
    # MobileNetV3
    mobilenet_model = MobileNetV3Model(num_classes=9)
    mobilenet_path = find_model_file(MODEL_DIR, 'mobilenetv3')
    if mobilenet_path and os.path.exists(mobilenet_path):
        checkpoint = torch.load(mobilenet_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            mobilenet_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            mobilenet_model.load_state_dict(checkpoint)
        mobilenet_model = mobilenet_model.to(device)
        mobilenet_model.eval()
        print(f"MobileNetV3 加载成功: {os.path.basename(mobilenet_path)}")
    else:
        print(f"警告: MobileNetV3模型文件不存在")
        return None, None, None
    
    # ResNet50
    resnet_model = ResNet50Model(num_classes=9)
    resnet_path = find_model_file(MODEL_DIR, 'resnet50')
    if resnet_path and os.path.exists(resnet_path):
        checkpoint = torch.load(resnet_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            resnet_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            resnet_model.load_state_dict(checkpoint)
        resnet_model = resnet_model.to(device)
        resnet_model.eval()
        print(f"ResNet50 加载成功: {os.path.basename(resnet_path)}")
    
    # CNN+Transformer
    cnn_transformer_model = CNNTransformerModel(num_classes=9)
    cnn_transformer_path = find_model_file(MODEL_DIR, 'cnn_transformer')
    if cnn_transformer_path and os.path.exists(cnn_transformer_path):
        checkpoint = torch.load(cnn_transformer_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            cnn_transformer_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            cnn_transformer_model.load_state_dict(checkpoint)
        cnn_transformer_model = cnn_transformer_model.to(device)
        cnn_transformer_model.eval()
        print(f"CNN+Transformer 加载成功: {os.path.basename(cnn_transformer_path)}")
    
    return mobilenet_model, resnet_model, cnn_transformer_model

def predict_models(models, test_loader):
    """使用三个模型对测试集进行预测"""
    mobilenet_model, resnet_model, cnn_transformer_model = models
    
    mobilenet_preds = []
    resnet_preds = []
    cnn_transformer_preds = []
    true_labels = []
    
    print("正在进行模型预测...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="预测进度"):
            images = images.to(device)
            
            # MobileNetV3预测
            if mobilenet_model is not None:
                mobilenet_output = mobilenet_model(images)
                mobilenet_pred = torch.argmax(mobilenet_output, dim=1).cpu().numpy()
                mobilenet_preds.extend(mobilenet_pred)
            
            # ResNet50预测
            if resnet_model is not None:
                resnet_output = resnet_model(images)
                resnet_pred = torch.argmax(resnet_output, dim=1).cpu().numpy()
                resnet_preds.extend(resnet_pred)
            
            # CNN+Transformer预测
            if cnn_transformer_model is not None:
                cnn_transformer_output = cnn_transformer_model(images)
                cnn_transformer_pred = torch.argmax(cnn_transformer_output, dim=1).cpu().numpy()
                cnn_transformer_preds.extend(cnn_transformer_pred)
            
            true_labels.extend(labels.numpy())
    
    return {
        'MobileNetV3': np.array(mobilenet_preds) if mobilenet_preds else None,
        'ResNet50': np.array(resnet_preds) if resnet_preds else None,
        'CNN+Transformer': np.array(cnn_transformer_preds) if cnn_transformer_preds else None,
        'True': np.array(true_labels)
    }

def mcnemar_test(y_true, pred1, pred2):
    """
    McNemar's test for comparing two classifiers
    
    Returns:
        n00: both wrong
        n01: model1 wrong, model2 correct
        n10: model1 correct, model2 wrong
        n11: both correct
        p_value: statistical significance
    """
    n00 = np.sum((pred1 != y_true) & (pred2 != y_true))
    n01 = np.sum((pred1 != y_true) & (pred2 == y_true))
    n10 = np.sum((pred1 == y_true) & (pred2 != y_true))
    n11 = np.sum((pred1 == y_true) & (pred2 == y_true))
    
    # McNemar's test statistic (with continuity correction)
    if n01 + n10 > 0:
        statistic = abs(n01 - n10 - 1) ** 2 / (n01 + n10)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(statistic, df=1)
    else:
        p_value = 1.0
    
    return n00, n01, n10, n11, p_value

def calculate_metrics(y_true, y_pred):
    """计算分类指标"""
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    return accuracy, f1, recall

def main():
    print("=" * 60)
    print("模型性能统计显著性检验 (McNemar's Test)")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载测试数据
    print("\n正在加载测试数据...")
    test_dataset = TestDataset(DATA_DIR, transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 加载模型
    models = load_models()
    if all(m is None for m in models):
        print("错误: 所有模型都无法加载!")
        return
    
    # 预测
    predictions = predict_models(models, test_loader)
    y_true = predictions['True']
    
    # 计算各模型指标
    print("\n" + "=" * 60)
    print("各模型性能指标")
    print("=" * 60)
    
    results = []
    for model_name in ['MobileNetV3', 'ResNet50', 'CNN+Transformer']:
        if predictions[model_name] is not None:
            acc, f1, recall = calculate_metrics(y_true, predictions[model_name])
            results.append({
                '模型': model_name,
                '准确率': f"{acc*100:.2f}%",
                'macro-F1': f"{f1*100:.2f}%",
                'macro-Recall': f"{recall*100:.2f}%"
            })
            print(f"{model_name}: 准确率={acc*100:.2f}%, macro-F1={f1*100:.2f}%, macro-Recall={recall*100:.2f}%")
    
    # McNemar's test两两比较
    print("\n" + "=" * 60)
    print("McNemar's Test 配对比较结果")
    print("=" * 60)
    
    model_pairs = [
        ('MobileNetV3', 'ResNet50'),
        ('MobileNetV3', 'CNN+Transformer'),
        ('ResNet50', 'CNN+Transformer')
    ]
    
    mcnemar_results = []
    
    for model1_name, model2_name in model_pairs:
        pred1 = predictions[model1_name]
        pred2 = predictions[model2_name]
        
        if pred1 is None or pred2 is None:
            continue
        
        n00, n01, n10, n11, p_value = mcnemar_test(y_true, pred1, pred2)
        
        # 判断显著性
        if p_value < 0.001:
            significance = "高度显著 (p<0.001)"
        elif p_value < 0.01:
            significance = "非常显著 (p<0.01)"
        elif p_value < 0.05:
            significance = "显著 (p<0.05)"
        else:
            significance = "不显著 (p≥0.05)"
        
        # 判断哪个模型更好
        acc1 = np.mean(pred1 == y_true)
        acc2 = np.mean(pred2 == y_true)
        if acc1 > acc2:
            better = model1_name
        else:
            better = model2_name
        
        mcnemar_results.append({
            '模型对比': f"{model1_name} vs {model2_name}",
            'n00 (两者错)': n00,
            'n01 (model1错)': n01,
            'n10 (model2错)': n10,
            'n11 (两者对)': n11,
            'p-value': f"{p_value:.6f}",
            '显著性': significance,
            '更优模型': better
        })
        
        print(f"\n{model1_name} vs {model2_name}:")
        print(f"  n00 (两者错): {n00}")
        print(f"  n01 ({model1_name}错, {model2_name}对): {n01}")
        print(f"  n10 ({model1_name}对, {model2_name}错): {n10}")
        print(f"  n11 (两者对): {n11}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  显著性: {significance}")
        print(f"  更优模型: {better}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    mcnemar_df = pd.DataFrame(mcnemar_results)
    
    results_path = os.path.join(OUTPUT_DIR, 'model_performance_metrics.csv')
    mcnemar_path = os.path.join(OUTPUT_DIR, 'mcnemar_test_results.csv')
    
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    mcnemar_df.to_csv(mcnemar_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("结果已保存")
    print("=" * 60)
    print(f"性能指标: {results_path}")
    print(f"McNemar检验: {mcnemar_path}")
    
    # 生成Markdown表格
    print("\n" + "=" * 60)
    print("Markdown格式结果")
    print("=" * 60)
    
    print("\n### 表1 各模型性能指标")
    print(results_df.to_markdown(index=False))
    
    print("\n### 表2 McNemar's Test配对比较结果")
    print(mcnemar_df.to_markdown(index=False))
    
    return results_df, mcnemar_df

if __name__ == "__main__":
    main()