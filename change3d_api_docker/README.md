# 遥感影像变化检测系统 - 后端API服务

## 项目简介

遥感影像变化检测系统后端是基于FastAPI和PyTorch构建的高性能AI推理服务，提供遥感影像变化检测的RESTful API接口。系统采用MVC架构设计，支持Docker容器化部署，具备强大的地理空间数据处理能力和深度学习推理性能。

## 技术栈

### 🧠 深度学习 & AI
- **PyTorch 2.8.0** - 核心深度学习框架
- **CUDA 12.6** - GPU并行计算加速
- **X3D模型** - 3D卷积神经网络架构
- **计算机视觉** - OpenCV、图像处理算法

### 🌍 地理信息系统 (GIS)
- **GDAL** - 地理空间数据抽象库
- **GeoPandas** - 地理空间数据分析
- **Rasterio** - 栅格影像处理
- **Fiona** - 矢量数据处理
- **Shapely** - 几何对象处理
- **PyProj** - 投影变换

### 🚀 后端开发
- **FastAPI** - 高性能异步Web框架
- **Uvicorn** - ASGI服务器
- **Pydantic** - 数据验证和序列化
- **异步编程** - 并发处理优化
- **MVC架构** - 模型-视图-控制器设计

### 🐳 容器化 & 部署
- **Docker** - 容器化部署
- **Docker Compose** - 多容器编排
- **NVIDIA Docker** - GPU容器支持
- **Ubuntu 20.04** - 基础操作系统

### 📊 数据处理
- **NumPy/SciPy** - 科学计算
- **多线程/多进程** - 并行计算优化
- **批处理** - 大规模数据处理
- **H5Py** - HDF5文件格式支持

## 系统架构

### 整体架构
```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   前端界面      │ ──────────────→ │   FastAPI服务   │
│  (PySide6)      │                │   (Uvicorn)     │
└─────────────────┘                └─────────────────┘
                                           │
                                           ▼
┌─────────────────┐                ┌─────────────────┐
│   Docker容器    │ ◄────────────── │   AI模型服务    │
│   (GPU支持)     │                │   (PyTorch)     │
└─────────────────┘                └─────────────────┘
```

### 目录结构
```
change3d_api_docker/
├── main.py                    # FastAPI主应用
├── change_detection_model.py  # AI模型服务封装
├── path_connector.py          # 路径连接器
├── run_api.py                 # API服务启动器
├── requirements.txt           # Python依赖
├── Dockerfile.optimized       # Docker镜像构建
├── docker-compose.optimized.yml # Docker编排配置
├── dev_start.sh               # 开发环境启动脚本
├── dev_restart.sh             # 开发环境重启脚本
├── dev_logs.sh                # 日志查看脚本
├── t1/                        # 前时相数据目录
├── t2/                        # 后时相数据目录
├── output/                    # 输出结果目录
└── test/                      # 测试数据目录
```

## 核心功能

### 🎯 变化检测能力
- **单图像处理** - 普通图像变化检测
- **栅格影像处理** - 地理信息保持的变化检测
- **批量处理** - 大规模影像对批量分析
- **矢量导出** - Shapefile、GeoJSON格式输出

### 🔧 处理模式
- **single_image** - 单张普通图像处理
- **single_raster** - 单张栅格影像处理
- **batch_image** - 批量普通图像处理
- **batch_raster** - 批量栅格影像处理

### 📈 性能特性
- **GPU加速** - CUDA并行计算
- **异步处理** - 后台任务队列
- **内存优化** - 智能内存管理
- **并发支持** - 多任务并行处理

## API接口文档

### 基础接口

#### 健康检查
```http
GET /health
```
**响应示例：**
```json
{
  "status": "ok",
  "version": "1.1"
}
```

#### 任务状态查询
```http
GET /tasks/{task_id}
```
**响应示例：**
```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "mode": "single_image",
  "message": "处理完成",
  "output_path": "/app/output/result.png"
}
```

### 变化检测接口

#### 单图像变化检测
```http
POST /detect/single_image
```
**请求体：**
```json
{
  "mode": "single_image",
  "before_path": "/app/t1/image1.png",
  "after_path": "/app/t2/image1.png",
  "output_path": "/app/output/result.png"
}
```

#### 栅格影像变化检测
```http
POST /detect/single_raster
```
**请求体：**
```json
{
  "mode": "single_raster",
  "before_path": "/app/t1/image1.tif",
  "after_path": "/app/t2/image1.tif",
  "output_path": "/app/output/result.tif"
}
```

#### 批量图像处理
```http
POST /detect/batch_image
```
**请求体：**
```json
{
  "mode": "batch_image",
  "before_path": "/app/t1/",
  "after_path": "/app/t2/",
  "output_path": "/app/output/"
}
```

#### 批量栅格处理
```http
POST /detect/batch_raster
```
**请求体：**
```json
{
  "mode": "batch_raster",
  "before_path": "/app/t1/",
  "after_path": "/app/t2/",
  "output_path": "/app/output/"
}
```

## 安装部署

### 环境要求
- **操作系统**: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+
- **Python**: 3.10+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **GPU**: NVIDIA GPU (推荐) + CUDA 12.6
- **内存**: 8GB+ (推荐16GB+)
- **存储**: 20GB+ 可用空间

### Docker部署 (推荐)

#### 1. 克隆项目
```bash
git clone <repository-url>
cd 遥感影像变化检测系统V1.1
```

#### 2. 构建并启动服务
```bash
# 进入API目录
cd change3d_api_docker

# 构建并启动服务
docker-compose -f docker-compose.optimized.yml up --build -d

# 查看服务状态
docker-compose -f docker-compose.optimized.yml ps

# 查看日志
docker-compose -f docker-compose.optimized.yml logs -f
```

#### 3. 验证部署
```bash
# 健康检查
curl http://localhost:8000/health

# 访问API文档
# 浏览器打开: http://localhost:8000/docs
```

### 开发环境部署

#### 1. 使用开发脚本
```bash
# 启动开发环境
./dev_start.sh

# 查看日志
./dev_logs.sh

# 重启服务
./dev_restart.sh
```

#### 2. 手动启动
```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务
python run_api.py --host 0.0.0.0 --port 8000 --reload
```

## 配置说明

### 环境变量
```bash
# GPU配置
CUDA_VISIBLE_DEVICES=0

# Python路径
PYTHONPATH=/app

# OpenMP配置
KMP_DUPLICATE_LIB_OK=TRUE

# 调试模式
DEBUG=1
LOG_LEVEL=DEBUG
```

### Docker配置
```yaml
# docker-compose.optimized.yml
services:
  change3d-api-optimized:
    ports:
      - "8000:8000"
    volumes:
      - ./t1:/app/change3d_api_docker/t1:rw
      - ./t2:/app/change3d_api_docker/t2:rw
      - ./output:/app/change3d_api_docker/output:rw
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 使用示例

### Python客户端示例
```python
import requests
import json

# API基础URL
API_BASE_URL = "http://localhost:8000"

# 单图像变化检测
def detect_single_image(before_path, after_path, output_path):
    url = f"{API_BASE_URL}/detect/single_image"
    payload = {
        "mode": "single_image",
        "before_path": before_path,
        "after_path": after_path,
        "output_path": output_path
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# 使用示例
result = detect_single_image(
    "/app/t1/image1.png",
    "/app/t2/image1.png", 
    "/app/output/result.png"
)
print(result)
```

### cURL示例
```bash
# 单图像变化检测
curl -X POST "http://localhost:8000/detect/single_image" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "single_image",
    "before_path": "/app/t1/image1.png",
    "after_path": "/app/t2/image1.png",
    "output_path": "/app/output/result.png"
  }'

# 查询任务状态
curl "http://localhost:8000/tasks/task_abc123"
```

## 性能优化

### GPU加速
- **CUDA支持**: 自动检测并使用GPU
- **内存优化**: 智能GPU内存管理
- **批处理**: 批量推理提升效率

### 并发处理
- **异步API**: FastAPI异步处理
- **后台任务**: BackgroundTasks支持
- **多线程**: 并行文件处理

### 内存管理
- **模型缓存**: 全局模型缓存
- **垃圾回收**: 自动内存清理
- **资源监控**: 实时资源使用监控

## 故障排除

### 常见问题

#### 1. GPU不可用
```bash
# 检查NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.6-base-ubuntu20.04 nvidia-smi

# 检查容器GPU访问
docker exec -it change3d-api-optimized nvidia-smi
```

#### 2. 内存不足
```bash
# 增加Docker内存限制
docker-compose -f docker-compose.optimized.yml up -d --scale change3d-api-optimized=1
```

#### 3. 端口冲突
```bash
# 修改端口映射
# 在docker-compose.optimized.yml中修改
ports:
  - "8001:8000"  # 使用8001端口
```

### 日志分析
```bash
# 查看实时日志
docker-compose -f docker-compose.optimized.yml logs -f

# 查看特定服务日志
docker-compose -f docker-compose.optimized.yml logs change3d-api-optimized

# 进入容器调试
docker exec -it change3d-api-optimized bash
```

## 开发指南

### 代码结构
- **模块化设计**: 功能模块独立封装
- **接口规范**: 统一的API接口定义
- **错误处理**: 完善的异常捕获机制

### 扩展开发
1. 在`change_detection_model.py`中添加新的处理模式
2. 在`main.py`中注册新的API端点
3. 更新Docker配置和依赖
4. 添加相应的测试用例

### 调试模式
```python
# 启用调试日志
logging.basicConfig(level=logging.DEBUG)

# 启用详细错误信息
import traceback
traceback.print_exc()
```

## 版本历史

### v1.1.0 (2024-01-01)
- 优化Docker镜像构建
- 增强GPU支持
- 改进错误处理机制
- 添加批量处理功能

### v1.0.0 (2023-12-01)
- 初始版本发布
- 基础变化检测功能
- FastAPI接口实现
- Docker容器化部署

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交代码更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

- 项目维护者：qianxiR
- 邮箱：support@rsiis.com
- 项目地址：https://github.com/your-repo/change3d-api

---

**注意**：使用本系统前请确保已正确配置GPU环境，并具备相应的遥感影像数据。 