# 遥感影像变化检测系统

本项目是一个基于深度学习的遥感影像变化检测系统，提供了多种变化检测算法和API接口。

## 系统特点

- 支持普通图像和栅格地理数据的变化检测
- 支持大型图像的滑动窗口处理
- 支持批量处理多对图像
- 提供REST API接口，方便集成
- 可导出矢量格式的变化边界（适用于地理数据）
- 支持Docker容器化部署，环境隔离，易于部署

## Docker 部署指南

### 环境要求

- Docker Desktop (Windows/Mac) 或 Docker Engine (Linux)
- Docker Compose
- NVIDIA Docker (可选，用于GPU加速)
- 至少 8GB 内存，20GB 磁盘空间

### 快速开始

#### 1. 克隆项目
```bash
git clone <repository-url>
cd 遥感影像变化检测系统V1.1
```

#### 2. 进入Docker目录
```bash
cd change3d_api_docker
```

#### 3. 构建和启动服务

**方法一：使用优化版本（推荐）**
```bash
# 构建并启动服务
docker-compose -f docker-compose.optimized.yml up --build -d

# 查看服务状态
docker-compose -f docker-compose.optimized.yml ps

# 查看日志
docker-compose -f docker-compose.optimized.yml logs -f
```

**方法二：使用部署脚本**
```bash
# 完整部署
./deploy.sh deploy

# 仅构建镜像
./deploy.sh build

# 启动服务
./deploy.sh start
```

**方法三：使用内存优化构建**
```bash
# 内存优化构建
./build_with_memory_optimization.sh
```

#### 4. 验证服务
```bash
# 健康检查
curl http://localhost:8000/health

# 访问API文档
# 浏览器打开: http://localhost:8000/docs
```

### Docker 配置说明

#### 镜像配置
- **基础镜像**: Ubuntu 20.04
- **Python环境**: Python 3.10 + Conda
- **深度学习框架**: PyTorch 2.8.0 + CUDA 12.6
- **Web框架**: FastAPI + Uvicorn
- **地理空间库**: GDAL, GeoPandas, Rasterio

#### 端口映射
- **API服务**: 8000:8000
- **数据目录**: 自动挂载到容器内

#### 数据目录结构
```
change3d_api_docker/
├── t1/                    # 前时相数据目录
├── t2/                    # 后时相数据目录
├── output/                # 输出结果目录
└── checkpoint/            # 模型检查点目录
```

### 开发环境配置

#### 代码热更新
Docker配置已支持代码热更新，修改代码后服务会自动重启：

```bash
# 查看实时日志
docker logs change3d-api-fixed -f

# 手动重启服务
docker restart change3d-api-fixed
```

#### 开发环境管理
```bash
# 启动开发环境
./dev_start.sh

# 查看开发日志
./dev_logs.sh

# 重启开发服务
./dev_restart.sh
```

### 生产环境部署

#### 1. 生产环境配置
```yaml
# docker-compose.prod.yml
version: '3'
services:
  change3d-api-prod:
    build:
      context: ..
      dockerfile: ./change3d_api_docker/Dockerfile.optimized
    image: change3d-api-prod:latest
    container_name: change3d-api-prod
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:rw
      - ./logs:/app/logs:rw
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
```

#### 2. 启动生产服务
```bash
# 生产环境部署
docker-compose -f docker-compose.prod.yml up -d

# 查看生产日志
docker-compose -f docker-compose.prod.yml logs -f
```

### 服务管理

#### 常用命令
```bash
# 查看服务状态
docker ps

# 查看服务日志
docker logs change3d-api-fixed

# 重启服务
docker restart change3d-api-fixed

# 停止服务
docker stop change3d-api-fixed

# 进入容器
docker exec -it change3d-api-fixed bash

# 清理资源
docker system prune -f
```

#### 故障排查
```bash
# 检查容器状态
docker ps -a

# 查看详细日志
docker logs change3d-api-fixed --tail 100

# 检查端口占用
netstat -tulpn | grep 8000

# 检查磁盘空间
docker system df
```

### 性能优化

#### 内存优化
- 使用 `--no-cache-dir` 减少pip缓存
- 分步安装地理空间库
- 定期清理Docker缓存

#### GPU加速
```bash
# 检查GPU支持
nvidia-smi

# 启用GPU加速
docker run --gpus all -p 8000:8000 change3d-api-optimized:latest
```

#### 网络优化
- 使用清华大学镜像源加速下载
- 配置Docker镜像加速器

### 监控和维护

#### 健康检查
```bash
# API健康检查
curl http://localhost:8000/health

# 容器健康检查
docker inspect change3d-api-fixed | grep Health
```

#### 日志管理
```bash
# 查看实时日志
docker logs -f change3d-api-fixed

# 导出日志
docker logs change3d-api-fixed > api.log

# 清理日志
docker system prune -f
```

#### 备份和恢复
```bash
# 备份镜像
docker save change3d-api-optimized:latest > change3d-api-backup.tar

# 恢复镜像
docker load < change3d-api-backup.tar

# 备份数据
tar -czf data-backup.tar.gz t1/ t2/ output/
```

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

#### Docker方式（推荐）
```bash
# 使用Docker启动
cd change3d_api_docker
docker-compose -f docker-compose.optimized.yml up -d

# 访问API文档
# http://localhost:8000/docs
```

#### 本地方式
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

### Docker环境下的API调用

```bash
# 上传图像到Docker容器
curl -X POST http://localhost:8000/detect \
  -F "before_path=@/path/to/before.png" \
  -F "after_path=@/path/to/after.png" \
  -F "mode=single_image"

# 批量处理
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "batch_image",
    "before_path": "/app/change3d_api_docker/t1",
    "after_path": "/app/change3d_api_docker/t2",
    "output_path": "/app/change3d_api_docker/output"
  }'
``` 