# 遥感影像变化检测系统API接口文档

## 一、API概述

### 1.1 基本信息
- **API版本**：V1.0
- **基础URL**：`http://localhost:8000`
- **协议**：HTTP/HTTPS
- **数据格式**：JSON、Multipart Form Data
- **认证方式**：无（开发版本）

### 1.2 响应格式
所有API响应均采用统一的JSON格式：
```json
{
    "status": "success|error",
    "message": "响应消息",
    "data": {
        // 响应数据
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

### 1.3 错误码说明
| 错误码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 401 | 未授权访问 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

## 二、变化检测接口

### 2.1 单图像变化检测

#### 2.1.1 接口信息
- **接口路径**：`/detect/single_image`
- **请求方法**：POST
- **Content-Type**：`multipart/form-data`

#### 2.1.2 请求参数
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| image1 | File | 是 | 前时相图像文件 |
| image2 | File | 是 | 后时相图像文件 |
| threshold | float | 否 | 检测阈值(0.0-1.0)，默认0.5 |
| output_format | string | 否 | 输出格式(png/jpg/tiff)，默认png |

#### 2.1.3 请求示例
```bash
curl -X POST "http://localhost:8000/detect/single_image" \
  -H "Content-Type: multipart/form-data" \
  -F "image1=@before.png" \
  -F "image2=@after.png" \
  -F "threshold=0.6" \
  -F "output_format=png"
```

#### 2.1.4 响应示例
```json
{
    "status": "success",
    "message": "变化检测完成",
    "data": {
        "result_path": "/app/results/change_detection_20250831_100000.png",
        "processing_time": 25.6,
        "change_statistics": {
            "changed_pixels": 12345,
            "total_pixels": 1048576,
            "change_percentage": 1.18,
            "confidence_score": 0.89
        },
        "file_size": "2.3MB",
        "dimensions": {
            "width": 1024,
            "height": 1024
        }
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

### 2.2 栅格影像变化检测

#### 2.2.1 接口信息
- **接口路径**：`/detect/single_raster`
- **请求方法**：POST
- **Content-Type**：`multipart/form-data`

#### 2.2.2 请求参数
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| raster1 | File | 是 | 前时相栅格文件 |
| raster2 | File | 是 | 后时相栅格文件 |
| threshold | float | 否 | 检测阈值(0.0-1.0)，默认0.5 |
| output_raster | boolean | 否 | 是否输出栅格结果，默认true |
| output_vector | boolean | 否 | 是否输出矢量结果，默认true |
| crs | string | 否 | 坐标参考系统，默认自动识别 |

#### 2.2.3 请求示例
```bash
curl -X POST "http://localhost:8000/detect/single_raster" \
  -H "Content-Type: multipart/form-data" \
  -F "raster1=@before.tif" \
  -F "raster2=@after.tif" \
  -F "threshold=0.6" \
  -F "output_raster=true" \
  -F "output_vector=true" \
  -F "crs=EPSG:4326"
```

#### 2.2.4 响应示例
```json
{
    "status": "success",
    "message": "栅格变化检测完成",
    "data": {
        "raster_result": {
            "path": "/app/results/change_detection_20250831_100000.tif",
            "format": "GeoTIFF",
            "size": "15.6MB"
        },
        "vector_result": {
            "path": "/app/results/change_detection_20250831_100000.shp",
            "format": "Shapefile",
            "size": "2.1MB",
            "feature_count": 156
        },
        "processing_time": 45.2,
        "geospatial_info": {
            "crs": "EPSG:4326",
            "bounds": [116.0, 39.0, 116.1, 39.1],
            "resolution": [0.0001, 0.0001],
            "dimensions": [1000, 1000]
        },
        "change_statistics": {
            "changed_pixels": 23456,
            "total_pixels": 1000000,
            "change_percentage": 2.35,
            "confidence_score": 0.92
        }
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

## 三、批量处理接口

### 3.1 批量图像处理

#### 3.1.1 接口信息
- **接口路径**：`/detect/batch_image`
- **请求方法**：POST
- **Content-Type**：`application/json`

#### 3.1.2 请求参数
```json
{
    "input_directory": "/path/to/input/directory",
    "output_directory": "/path/to/output/directory",
    "file_pattern": "*.png",
    "threshold": 0.5,
    "output_format": "png",
    "max_workers": 4,
    "batch_size": 8
}
```

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| input_directory | string | 是 | 输入目录路径 |
| output_directory | string | 是 | 输出目录路径 |
| file_pattern | string | 否 | 文件匹配模式，默认"*.png" |
| threshold | float | 否 | 检测阈值，默认0.5 |
| output_format | string | 否 | 输出格式，默认"png" |
| max_workers | int | 否 | 最大工作线程数，默认4 |
| batch_size | int | 否 | 批次大小，默认8 |

#### 3.1.3 请求示例
```bash
curl -X POST "http://localhost:8000/detect/batch_image" \
  -H "Content-Type: application/json" \
  -d '{
    "input_directory": "/app/data/input",
    "output_directory": "/app/data/output",
    "file_pattern": "*.png",
    "threshold": 0.6,
    "output_format": "png",
    "max_workers": 4
  }'
```

#### 3.1.4 响应示例
```json
{
    "status": "success",
    "message": "批量处理任务已启动",
    "data": {
        "task_id": "batch_20250831_100000",
        "total_files": 50,
        "estimated_time": 1200,
        "progress_url": "/task/progress/batch_20250831_100000",
        "status": "processing"
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

### 3.2 批量栅格处理

#### 3.2.1 接口信息
- **接口路径**：`/detect/batch_raster`
- **请求方法**：POST
- **Content-Type**：`application/json`

#### 3.2.2 请求参数
```json
{
    "input_directory": "/path/to/input/directory",
    "output_directory": "/path/to/output/directory",
    "file_pattern": "*.tif",
    "threshold": 0.5,
    "output_raster": true,
    "output_vector": true,
    "crs": "EPSG:4326",
    "max_workers": 2,
    "batch_size": 4
}
```

#### 3.2.3 响应示例
```json
{
    "status": "success",
    "message": "批量栅格处理任务已启动",
    "data": {
        "task_id": "batch_raster_20250831_100000",
        "total_files": 20,
        "estimated_time": 1800,
        "progress_url": "/task/progress/batch_raster_20250831_100000",
        "status": "processing"
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

## 四、任务管理接口

### 4.1 获取任务进度

#### 4.1.1 接口信息
- **接口路径**：`/task/progress/{task_id}`
- **请求方法**：GET

#### 4.1.2 请求示例
```bash
curl -X GET "http://localhost:8000/task/progress/batch_20250831_100000"
```

#### 4.1.3 响应示例
```json
{
    "status": "success",
    "message": "任务进度查询成功",
    "data": {
        "task_id": "batch_20250831_100000",
        "status": "processing",
        "progress": {
            "completed": 25,
            "total": 50,
            "percentage": 50.0,
            "current_file": "image_025.png"
        },
        "statistics": {
            "start_time": "2025-08-31T10:00:00Z",
            "elapsed_time": 600,
            "estimated_remaining": 600,
            "processing_speed": "2.5 files/minute"
        },
        "results": {
            "successful": 25,
            "failed": 0,
            "error_files": []
        }
    },
    "timestamp": "2025-08-31T10:10:00Z"
}
```

### 4.2 取消任务

#### 4.2.1 接口信息
- **接口路径**：`/task/cancel/{task_id}`
- **请求方法**：POST

#### 4.2.2 请求示例
```bash
curl -X POST "http://localhost:8000/task/cancel/batch_20250831_100000"
```

#### 4.2.3 响应示例
```json
{
    "status": "success",
    "message": "任务已取消",
    "data": {
        "task_id": "batch_20250831_100000",
        "status": "cancelled",
        "completed_files": 25,
        "cancelled_at": "2025-08-31T10:10:00Z"
    },
    "timestamp": "2025-08-31T10:10:00Z"
}
```

## 五、系统管理接口

### 5.1 系统状态查询

#### 5.1.1 接口信息
- **接口路径**：`/system/status`
- **请求方法**：GET

#### 5.1.2 响应示例
```json
{
    "status": "success",
    "message": "系统状态查询成功",
    "data": {
        "system_info": {
            "version": "V1.0",
            "uptime": 86400,
            "start_time": "2025-08-30T10:00:00Z"
        },
        "hardware_info": {
            "cpu_usage": 45.2,
            "memory_usage": 68.5,
            "gpu_usage": 75.3,
            "disk_usage": 45.8
        },
        "model_info": {
            "model_loaded": true,
            "model_version": "X3D_V1.0",
            "gpu_available": true,
            "cuda_version": "12.6"
        },
        "task_info": {
            "active_tasks": 2,
            "queued_tasks": 5,
            "completed_tasks": 156
        }
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

### 5.2 模型信息查询

#### 5.2.1 接口信息
- **接口路径**：`/system/model_info`
- **请求方法**：GET

#### 5.2.2 响应示例
```json
{
    "status": "success",
    "message": "模型信息查询成功",
    "data": {
        "model_name": "X3D_ChangeDetection",
        "model_version": "V1.0",
        "model_path": "/app/models/x3d_model.pth",
        "model_size": "256MB",
        "architecture": {
            "backbone": "X3D",
            "input_channels": 6,
            "output_channels": 1,
            "feature_dim": 2048
        },
        "performance": {
            "inference_time": "25.6ms",
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94
        },
        "supported_formats": ["png", "jpg", "tiff", "geotiff"],
        "max_input_size": [4096, 4096]
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

## 六、错误处理

### 6.1 错误响应格式
```json
{
    "status": "error",
    "message": "错误描述",
    "error_code": "ERROR_CODE",
    "details": {
        "field": "具体错误字段",
        "reason": "错误原因"
    },
    "timestamp": "2025-08-31T10:00:00Z"
}
```

### 6.2 常见错误码
| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| INVALID_FILE_FORMAT | 不支持的文件格式 | 检查文件格式是否支持 |
| FILE_TOO_LARGE | 文件过大 | 压缩文件或使用分块处理 |
| INVALID_PARAMETER | 参数错误 | 检查请求参数格式 |
| MODEL_NOT_LOADED | 模型未加载 | 重启服务或检查模型文件 |
| GPU_NOT_AVAILABLE | GPU不可用 | 检查GPU驱动和CUDA安装 |
| INSUFFICIENT_MEMORY | 内存不足 | 减少批次大小或关闭其他程序 |
| TASK_NOT_FOUND | 任务不存在 | 检查任务ID是否正确 |
| PROCESSING_FAILED | 处理失败 | 查看详细错误日志 |

## 七、使用示例

### 7.1 Python客户端示例
```python
import requests
import json

class ChangeDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def detect_single_image(self, image1_path, image2_path, threshold=0.5):
        """单图像变化检测"""
        url = f"{self.base_url}/detect/single_image"
        
        with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
            files = {
                'image1': f1,
                'image2': f2
            }
            data = {
                'threshold': threshold,
                'output_format': 'png'
            }
            
            response = requests.post(url, files=files, data=data)
            return response.json()
    
    def detect_single_raster(self, raster1_path, raster2_path, threshold=0.5):
        """栅格影像变化检测"""
        url = f"{self.base_url}/detect/single_raster"
        
        with open(raster1_path, 'rb') as f1, open(raster2_path, 'rb') as f2:
            files = {
                'raster1': f1,
                'raster2': f2
            }
            data = {
                'threshold': threshold,
                'output_raster': True,
                'output_vector': True
            }
            
            response = requests.post(url, files=files, data=data)
            return response.json()
    
    def batch_process(self, input_dir, output_dir, file_pattern="*.png"):
        """批量处理"""
        url = f"{self.base_url}/detect/batch_image"
        
        data = {
            'input_directory': input_dir,
            'output_directory': output_dir,
            'file_pattern': file_pattern,
            'threshold': 0.5,
            'max_workers': 4
        }
        
        response = requests.post(url, json=data)
        return response.json()
    
    def get_task_progress(self, task_id):
        """获取任务进度"""
        url = f"{self.base_url}/task/progress/{task_id}"
        response = requests.get(url)
        return response.json()

# 使用示例
client = ChangeDetectionClient()

# 单图像检测
result = client.detect_single_image("before.png", "after.png", threshold=0.6)
print(f"检测完成，结果保存在: {result['data']['result_path']}")

# 批量处理
task = client.batch_process("/input", "/output", "*.png")
task_id = task['data']['task_id']

# 监控进度
while True:
    progress = client.get_task_progress(task_id)
    print(f"进度: {progress['data']['progress']['percentage']}%")
    if progress['data']['status'] in ['completed', 'failed', 'cancelled']:
        break
```

### 7.2 JavaScript客户端示例
```javascript
class ChangeDetectionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async detectSingleImage(image1File, image2File, threshold = 0.5) {
        const formData = new FormData();
        formData.append('image1', image1File);
        formData.append('image2', image2File);
        formData.append('threshold', threshold);
        formData.append('output_format', 'png');
        
        const response = await fetch(`${this.baseUrl}/detect/single_image`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    async batchProcess(inputDir, outputDir, filePattern = '*.png') {
        const data = {
            input_directory: inputDir,
            output_directory: outputDir,
            file_pattern: filePattern,
            threshold: 0.5,
            max_workers: 4
        };
        
        const response = await fetch(`${this.baseUrl}/detect/batch_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        return await response.json();
    }
    
    async getTaskProgress(taskId) {
        const response = await fetch(`${this.baseUrl}/task/progress/${taskId}`);
        return await response.json();
    }
}

// 使用示例
const client = new ChangeDetectionClient();

// 文件上传处理
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const image1File = document.getElementById('image1').files[0];
    const image2File = document.getElementById('image2').files[0];
    
    try {
        const result = await client.detectSingleImage(image1File, image2File, 0.6);
        console.log('检测完成:', result);
    } catch (error) {
        console.error('检测失败:', error);
    }
});
```

---

**文档版本**：V1.0
**创建日期**：2025年8月31日
**更新日期**：2025年8月31日
**技术支持**：support@rsiis.com
