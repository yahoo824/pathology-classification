#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病理组织分类后端服务
功能：接收前端上传的病理切片图片，使用训练好的模型进行分类，支持模型选择
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
MODEL_DIR = r"D:\python_project\BS\model_checkpoints\new_models"
IMAGE_SIZE = (224, 224)

# 数据变换
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ResNet50模型定义
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.resnet50 = models.resnet50(weights=None)
        in_features = self.resnet50.fc.in_features
        # 添加Dropout层，与训练模型结构保持一致
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet50(x)

# MobileNetV3模型定义
class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.mobilenet(x)

# CNN+Transformer模型定义
class PatchEmbedding(nn.Module):
    """固定适配：输入2048通道 + 7×7尺寸，patch_size=2 → 输出3×3序列"""
    def __init__(self):
        super().__init__()
        # 核心：2048输入通道 + 2×2卷积核 → 完美匹配7×7输入，输出3×3特征
        self.proj = nn.Conv2d(2048, 512, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.proj(x)  # [B,2048,7,7] → [B,512,3,3]
        x = x.flatten(2)  # [B,512,9]
        x = x.transpose(1, 2)  # [B,9,512]
        return x


class TransformerEncoder(nn.Module):
    """轻量化编码器：适配GPU，提速训练"""
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CNNTransformerModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # 固定适配：ResNet50完整主干，输出2048通道+7×7尺寸
        self.cnn_backbone = models.resnet50(weights=None)
        # 移除最后的全局池化和分类层，保留特征提取部分
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])

        # 加载适配好的分块嵌入层
        self.patch_embed = PatchEmbedding()
        self.transformer_encoder = TransformerEncoder(512, 4)

        # 分类头：适配9分类任务
        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Step1: CNN提取特征 → [B,2048,7,7]
        cnn_feat = self.cnn_backbone(x)
        # Step2: 分块嵌入 → [B,9,512]
        patch_feat = self.patch_embed(cnn_feat)
        # Step3: Transformer全局特征建模
        trans_feat = self.transformer_encoder(patch_feat)
        # Step4: 全局池化+分类
        global_feat = trans_feat.mean(dim=1)
        out = self.fc(global_feat)
        return out

# 加载模型
def load_model(model_type):
    """加载指定类型的模型"""
    if model_type == "resnet50":
        model = ResNet50Model().to(DEVICE)
        model_path = os.path.join(MODEL_DIR, "resnet50_best_20260415_003939.pth")
    elif model_type == "mobilenetv3":
        model = MobileNetV3Model().to(DEVICE)
        model_path = os.path.join(MODEL_DIR, "mobilenetv3_best_20260414_195234.pth")
    elif model_type == "cnn_transformer":
        model = CNNTransformerModel().to(DEVICE)
        model_path = os.path.join(MODEL_DIR, "cnn_transformer_best_20260414_202158.pth")
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# 预测函数
def predict_image(model, image):
    """预测单张图片"""
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    class_name = CLASS_NAMES[pred]
    confidence = prob[pred]
    
    return {
        "class_name": class_name,
        "confidence": float(confidence),
        "probabilities": {CLASS_NAMES[i]: float(prob[i]) for i in range(len(CLASS_NAMES))}
    }

# 创建Flask应用
app = Flask(__name__, static_folder='frontend')
CORS(app)  # 允许跨域请求

# 预加载所有模型
print("正在加载模型...")
models_dict = {
    "resnet50": load_model("resnet50"),
    "mobilenetv3": load_model("mobilenetv3"),
    "cnn_transformer": load_model("cnn_transformer")
}
print(f"所有模型加载成功！使用设备: {DEVICE}")

# 路由：首页
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

# 路由：预测API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({"error": "没有上传文件"}), 400
        
        file = request.files['image']
        
        # 检查文件类型
        if file.filename == '':
            return jsonify({"error": "文件名不能为空"}), 400
        
        # 获取模型类型
        model_type = request.form.get('model', 'mobilenetv3')
        if model_type not in models_dict:
            return jsonify({"error": f"未知模型类型: {model_type}"}), 400
        
        # 读取图片
        image = Image.open(file.stream).convert('RGB')
        
        # 获取模型并预测
        model = models_dict[model_type]
        result = predict_image(model, image)
        
        # 返回结果
        return jsonify(result)
        
    except Exception as e:
        print(f"预测出错: {str(e)}")
        return jsonify({"error": f"预测失败: {str(e)}"}), 500

# 路由：静态文件
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)

# 路由：样本文件

@app.route('/sample/<path:filename>')
def serve_sample(filename):
    return send_from_directory('sample', filename)

if __name__ == '__main__':
    print("病理组织分类服务启动中...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=False)
