# 遥感影像变化检测系统

本项目是一个基于深度学习的遥感影像变化检测系统，提供了多种变化检测算法和API接口。

## 系统特点

- 支持普通图像和栅格地理数据的变化检测
- 支持大型图像的滑动窗口处理
- 支持批量处理多对图像
- 提供REST API接口，方便集成
- 可导出矢量格式的变化边界（适用于地理数据）

## 新的包结构

为了解决导入路径问题，系统已经重构为标准的Python包结构。

### 安装

在开发环境中，可以通过以下命令安装：

```bash
# 进入项目根目录
cd change3d

# 以开发模式安装
pip install -e .
```

### 目录结构

```
change3d/
├── .project_root          # 项目根目录标记文件
├── __init__.py            # 包初始化文件
├── paths.py               # 路径管理模块
├── setup.py               # 包安装配置
├── model/                 # 模型定义
├── data/                  # 数据处理
├── scripts_app/           # 算法实现层
│   ├── __init__.py
│   ├── batch_image_BCD.py
│   ├── large_image_BCD.py
│   ├── batch_raster_BCD.py
│   ├── large_raster_BCD.py
│   └── merge_vectors.py
└── change_detection_api/  # API服务层
    ├── __init__.py
    ├── main.py
    └── scripts/           # API脚本层
        ├── __init__.py
        ├── batch_image_api.py
        ├── large_image_api.py
        ├── batch_raster_api.py
        └── large_raster_api.py
```

## 如何使用

### 启动API服务

```bash
# 进入API目录
cd change3d/change_detection_api

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

然后访问 http://localhost:8000/docs 查看API文档。

### 导入模块

现在可以通过标准的包导入方式导入模块：

```python
# 导入路径管理模块
from change3d.paths import setup_module_paths

# 导入算法模块
from change3d.scripts_app.large_image_BCD import process_large_image

# 导入API服务
from change3d.change_detection_api.main import app
```

## 解决路径问题

如果仍然遇到导入问题，可以使用路径模块：

```python
# 导入路径模块并设置路径
from change3d.paths import setup_module_paths
paths = setup_module_paths()

# 获取重要路径
project_root = paths["project_root"]
scripts_app_dir = paths["scripts_app_dir"]
```

## 示例

### 使用API处理单张图像

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "pre_image=@/path/to/before.png" \
  -F "post_image=@/path/to/after.png" \
  -F "model_id=large_image_api"
```

### 批量处理目录中的图像

```bash
curl -X POST http://localhost:8000/api/v1/batch_detect \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "batch_image_api",
    "source_type": "directory",
    "pre_dir": "/path/to/before",
    "post_dir": "/path/to/after",
    "output_dir": "/path/to/output"
  }'
``` 