# 基于深度学习的病理组织分类系统

## 项目简介

本项目是一个基于深度学习的病理组织分类系统，用于对病理切片图像进行自动分类。系统包含一个简洁的前端界面和一个 Flask 后端服务，用户可以上传组织图像并选择不同的模型进行推理，最终得到分类结果及置信度。

该项目适用于学习、实验和演示场景，重点展示了从数据处理、模型训练、权重加载到部署推理的一整套流程。

## 功能特点

- 支持上传病理组织图像进行推理
- 支持切换不同模型进行分类
- 返回分类类别、置信度和各类别概率
- 提供一个简单易用的 Web 前端界面
- 支持基于 PyTorch 的深度学习模型推理

## 支持的模型

项目中提供了以下三种模型：

- ResNet50
- MobileNetV3
- CNN + Transformer

对应的训练权重文件位于 [BS/model_checkpoints](BS/model_checkpoints) 目录中。

## 项目结构

```text
.
├── BS/
│   ├── backend_server.py          # Flask 后端服务
│   ├── frontend/                  # 前端静态页面
│   ├── model_checkpoints/         # 模型权重与训练日志
│   ├── config.py                  # 配置文件
│   ├── data_loader.py             # 数据加载脚本
│   ├── generate_labels.py         # 标签生成脚本
│   ├── model_resnet50.py          # ResNet50 模型定义
│   ├── model_mobilenetv3.py       # MobileNetV3 模型定义
│   ├── model_cnn_transformer.py   # CNN + Transformer 模型定义
│   └── ...                        # 其他训练与分析脚本
└── README.md
```

## 环境要求

- Python 3.10+
- PyTorch
- TorchVision
- Flask
- Flask-CORS
- Pillow
- NumPy

建议使用 Conda 环境管理 Python 依赖。

## 安装步骤

### 1. 创建并激活虚拟环境（可选，但推荐）

```bash
conda create -n cdtu python=3.10
conda activate cdtu
```

### 2. 安装依赖

```bash
pip install torch torchvision flask flask-cors pillow numpy
```

如果你使用的是 Windows 系统且需要 GPU 加速，请确保已安装与当前 PyTorch 版本兼容的 CUDA 运行环境。

## 如何运行

### 1. 启动后端服务

在项目根目录执行：

```bash
python BS/backend_server.py
```

启动成功后，终端中会显示类似以下信息：

```text
* Running on http://127.0.0.1:5000
```

### 2. 打开前端页面

在浏览器中访问：

```text
http://127.0.0.1:5000
```

## 使用方法

1. 打开浏览器访问首页。
2. 上传一张病理组织图像。
3. 选择所需的模型类型。
4. 点击预测按钮。
5. 系统将返回分类结果、置信度和各类别概率。

## API 接口说明

### 预测接口

- 路径：`/predict`
- 方法：`POST`
- 参数：
  - `image`：上传的图片文件
  - `model`：模型类型，可选值为 `resnet50`、`mobilenetv3` 或 `cnn_transformer`


## 训练与模型文件说明

训练后的模型权重和日志文件保存在 [BS/model_checkpoints](BS/model_checkpoints) 目录中。若将项目上传到 GitHub，建议不要直接提交大尺寸模型文件和训练数据，以免造成仓库过大。

## 注意事项

- 本项目默认使用 GPU（如果可用），否则会自动切换到 CPU。
- 模型加载时可能出现 PyTorch 的未来兼容警告，但这通常不影响当前推理运行。
- 由于模型权重文件较大，实际使用前请确认对应权重文件已经存在。
- 该项目更适合作为学习、研究和演示用途，而不是生产级部署方案。

## 许可证

本项目仅供学习、研究和教学使用。
