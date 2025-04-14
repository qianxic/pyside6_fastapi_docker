from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from enum import Enum
from typing import Optional, Dict, Any, List
import os
# import tempfile # Not used
from datetime import datetime
import uuid
# import json # Not used
from fastapi.responses import RedirectResponse, Response
# import shutil # Not used
# import time # Not used
import asyncio
import logging

# 直接导入模型和模式
# import torch # Not used directly here
# from change3d_docker.scripts_app.large_image_BCD import process_and_save as process_and_save_image # Not used directly here
# from change3d_docker.scripts_app.large_raster_BCD import process_and_save as process_and_save_raster # Not used directly here
from change3d_api_docker.change_detection_model import detection_model, DetectionMode
# from change3d_docker.scripts_app.batch_image_BCD import process_and_save as process_and_save_batch_image # Not used directly here
# from change3d_docker.scripts_app.batch_raster_BCD import process_and_save as process_and_save_batch_raster # Not used directly here
# 创建全局变量
_detection_model = detection_model
_DetectionMode = DetectionMode

'''
uvicorn main:app --reload
http://127.0.0.1:8000/docs

'''


# 启动 FastAPI 应用
app = FastAPI(title="遥感影像变化检测API", description="支持单图像、单影像、多图像、多影像变化检测")

# 添加根路径处理程序，将访问重定向到文档页面
@app.get("/")
def read_root():
    """将根路径访问重定向到API文档页面"""
    return RedirectResponse(url="/docs")

# 添加favicon.ico处理程序
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """返回一个空响应，避免404日志"""
    return Response(content="", media_type="image/x-icon")

# 处理模式枚举
class ProcessingMode(str, Enum):
    single_image = "single_image"         # 普通图像单张处理
    single_raster = "single_raster"       # 栅格影像单张处理
    batch_image = "batch_image"           # 普通图像批处理
    batch_raster = "batch_raster"         # 栅格影像批处理

# 请求体数据模型
class PathInput(BaseModel):
    mode: ProcessingMode                  # 处理模式
    before_path: Optional[str] = None    # 前时相路径（或文件夹） - Docker 内部路径
    after_path: Optional[str] = None     # 后时相路径（或文件夹） - Docker 内部路径
    output_path: Optional[str] = None    # 输出路径（可选） - Docker 内部期望路径

# 检测任务响应模型
class DetectionResponse(BaseModel):
    task_id: str                          # 任务ID
    status: str                           # 任务状态
    mode: str                             # 处理模式
    message: str                          # 状态消息
    output_path: Optional[str] = None     # 输出路径

# 任务状态详情模型
class TaskStatusDetail(BaseModel):
    task_id: str                          # 任务ID
    status: str                           # 任务状态
    mode: str                             # 处理模式
    message: str                          # 状态消息
    start_time: Optional[str] = None      # 开始时间
    end_time: Optional[str] = None        # 结束时间
    output_path: Optional[str] = None     # 输出路径
    result: Optional[Dict[str, Any]] = None  # 处理结果

# 健康检查端点
@app.get("/health")
def health_check():
    """健康检查接口，用于检测API服务是否正常运行"""
    return {"status": "ok", "version": "1.1"}

# 任务状态跟踪
detection_tasks: Dict[str, Dict[str, Any]] = {}

# 定义原始路径输入模型
class OriginalOutputPathInput(BaseModel):
    original_user_output_path: str  # 用户的原始本机输出路径

@app.put("/tasks/{task_id}/original_output", status_code=200, summary="设置任务的原始输出路径")
async def set_task_original_output(task_id: str, data: OriginalOutputPathInput):
    """
    允许外部调用者为已创建的任务设置其对应的用户原始输出路径。
    这通常在 /detect/... 调用成功返回 task_id 后立即调用。
    
    ** 此API已废弃，复制操作将由外部进程完成 **
    """
    return {"message": "此API已废弃，复制操作将由外部进程完成"}

@app.get("/history")
def get_request_history():
    """获取最近的请求历史"""
    return {"history": request_history[-10:]}  # 最近10条

# 接口主函数 - 路径处理
@app.post("/process/")
async def process_paths(data: PathInput):
    """处理路径接口"""
    # 这里只是简单返回，实际应用中可能不需要这个端点或需要更复杂的逻辑
    return {
        "mode": data.mode,
        "before_path": data.before_path,
        "after_path": data.after_path,
        "output_path": data.output_path
    }

# 添加变化检测接口
@app.post("/detect/single_image", response_model=DetectionResponse)
async def detect_changes(data: PathInput, background_tasks: BackgroundTasks):
    """
    执行普通图像变化检测
    
    将执行单对普通图像的变化检测，支持常见图像格式如JPG、PNG等，
    输出图像级别的变化检测结果
    
    - **mode**: 必须设置为 `single_image`
    - **before_path**: 前时相图像路径
    - **after_path**: 后时相图像路径
    - **output_path**: 输出结果路径
    """
    # 强制设置模式为单图像处理，确保即使请求中的模式不正确，也能正确处理
    data.mode = ProcessingMode.single_image
    
    # 生成任务ID
    task_id = f"task_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 准备传递给后台任务的路径信息
    processed_paths = {
        "output_path": data.output_path # Docker 内部期望路径
    }
    
    # 初始化任务状态
    detection_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "变化检测任务已创建，等待执行",
        "before_path": data.before_path, # Docker 路径
        "after_path": data.after_path,   # Docker 路径
        "output_path": data.output_path, # Docker 期望路径
    }
    
    # 在后台执行任务，传递 processed_paths
    background_tasks.add_task(
        run_detection_task,
        task_id,
        data.mode,
        data.before_path, # Docker 路径
        data.after_path,  # Docker 路径
        processed_paths # 只包含 output_path
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "变化检测任务已创建，等待执行",
        "output_path": data.output_path
    }

@app.post("/detect/single_raster", response_model=DetectionResponse)
async def detect_raster_changes(data: PathInput, background_tasks: BackgroundTasks):
    """
    执行栅格影像变化检测
    
    将执行单张栅格影像的变化检测，支持带有地理信息的栅格数据（如GeoTIFF），
    并可导出矢量结果（Shapefile和GeoJSON格式）
    
    - **mode**: 必须设置为 `single_raster`
    - **before_path**: 前时相栅格影像路径
    - **after_path**: 后时相栅格影像路径
    - **output_path**: 输出结果路径
    """
    # 强制设置模式为栅格处理，确保即使请求中的模式不正确，也能正确处理
    data.mode = ProcessingMode.single_raster
    
    # 生成任务ID
    task_id = f"task_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 准备传递给后台任务的路径信息
    processed_paths = {
        "output_path": data.output_path
    }
    
    # 初始化任务状态
    detection_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "栅格影像变化检测任务已创建，等待执行",
        "before_path": data.before_path,
        "after_path": data.after_path,
        "output_path": data.output_path,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "result": None
    }
    
    # 在后台执行任务，传递 processed_paths
    background_tasks.add_task(
        run_detection_task,
        task_id,
        data.mode,
        data.before_path,
        data.after_path,
        processed_paths
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "栅格影像变化检测任务已创建，等待执行",
        "output_path": data.output_path
    }

@app.post("/detect/batch_image", response_model=DetectionResponse)
async def detect_batch_images(data: PathInput, background_tasks: BackgroundTasks):
    """
    执行批量图像变化检测
    
    将执行批量普通图像的变化检测，支持常见图像格式如JPG、PNG等，
    在指定的目录中查找匹配的图像对进行变化检测
    
    - **mode**: 必须设置为 `batch_image`
    - **before_path**: 前时相图像目录
    - **after_path**: 后时相图像目录
    - **output_path**: 输出结果目录
    """
    # 强制设置模式为批量图像处理
    data.mode = ProcessingMode.batch_image
    
    # 生成任务ID
    task_id = f"task_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 准备传递给后台任务的路径信息
    processed_paths = {
        "output_path": data.output_path # Docker 内部目录
    }
    
    # 初始化任务状态
    detection_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "批量图像变化检测任务已创建，等待执行",
        "before_path": data.before_path,
        "after_path": data.after_path,
        "output_path": data.output_path,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "result": None
    }
    
    # 在后台执行任务，传递 processed_paths
    background_tasks.add_task(
        run_detection_task,
        task_id,
        data.mode,
        data.before_path,
        data.after_path,
        processed_paths
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "批量图像变化检测任务已创建，等待执行",
        "output_path": data.output_path
    }

@app.post("/detect/batch_raster", response_model=DetectionResponse)
async def detect_batch_rasters(data: PathInput, background_tasks: BackgroundTasks):
    """
    执行批量栅格影像变化检测
    
    将执行批量栅格影像的变化检测，支持带有地理信息的栅格数据（如GeoTIFF），
    在指定的目录中查找匹配的影像对进行变化检测并导出矢量结果
    
    - **mode**: 必须设置为 `batch_raster`
    - **before_path**: 前时相栅格影像目录
    - **after_path**: 后时相栅格影像目录
    - **output_path**: 输出结果目录
    """
    # 强制设置模式为批量栅格处理
    data.mode = ProcessingMode.batch_raster
    
    # 生成任务ID
    task_id = f"task_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 准备传递给后台任务的路径信息
    processed_paths = {
        "output_path": data.output_path # Docker 内部目录
    }
    
    # 初始化任务状态
    detection_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "批量栅格影像变化检测任务已创建，等待执行",
        "before_path": data.before_path,
        "after_path": data.after_path,
        "output_path": data.output_path,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "result": None
    }
    
    # 在后台执行任务，传递 processed_paths
    background_tasks.add_task(
        run_detection_task,
        task_id,
        data.mode,
        data.before_path,
        data.after_path,
        processed_paths
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "mode": data.mode,
        "message": "批量栅格影像变化检测任务已创建，等待执行",
        "output_path": data.output_path
    }


async def run_detection_task(task_id: str, mode_str: str, before_path: str, \
                           after_path: str, processed_paths: dict):
    """在后台执行变化检测任务 (检查文件存在后设置 running 状态)"""
    if task_id not in detection_tasks:
        return

    # 1. 检查文件在容器内是否存在 (带短暂延迟)
    try:
        await asyncio.sleep(0.8) # 增加一点延迟以确保文件系统同步
        before_exists = os.path.exists(before_path)
        after_exists = os.path.exists(after_path)

        if not before_exists or not after_exists:
            error_msg = f"任务 {task_id} 失败：输入文件在容器内未找到。\
Before ('{before_path}'): {before_exists}, After ('{after_path}'): {after_exists}"
            logging.error(error_msg)
            detection_tasks[task_id]["status"] = "failed"
            detection_tasks[task_id]["message"] = error_msg
            detection_tasks[task_id]["end_time"] = datetime.now().isoformat()
            return
    except Exception as check_e:
        error_msg = f"任务 {task_id} 失败：检查文件存在性时出错: {str(check_e)}"
        logging.error(error_msg, exc_info=True)
        detection_tasks[task_id]["status"] = "failed"
        detection_tasks[task_id]["message"] = error_msg
        detection_tasks[task_id]["end_time"] = datetime.now().isoformat()
        return

    # 2. 文件存在，明确设置状态为 running
    try:
        detection_tasks[task_id]["status"] = "running"
        detection_tasks[task_id]["message"] = "输入文件已确认，正在执行变化检测..."
    except KeyError:
        return

    docker_internal_output_path = processed_paths.get("output_path")
    model_result_data = {}

    # 3. 执行模型检测
    try:
        model_result_data = _detection_model.run_detection(
            mode=mode_str,
            before_path=before_path,
            after_path=after_path,
            output_path=docker_internal_output_path
        )

        if model_result_data.get("status") != "success":
            error_msg = model_result_data.get('message', f"模型处理失败，模式: {mode_str}")
            raise Exception(error_msg)

        # 模型处理成功，准备完整的结果数据
        all_output_files = []
        if 'output_path' in model_result_data and model_result_data['output_path']:
            all_output_files.append(model_result_data['output_path'])
        if 'quad_view_path' in model_result_data and model_result_data['quad_view_path']:
            all_output_files.append(model_result_data['quad_view_path'])
        if 'vector_files' in model_result_data and model_result_data['vector_files']:
            all_output_files.extend(model_result_data['vector_files'])
            for vector_file in model_result_data['vector_files']:
                if vector_file.endswith('.shp'):
                    base_dir = os.path.dirname(vector_file)
                    base_name = os.path.splitext(os.path.basename(vector_file))[0]
                    for ext in ['.dbf', '.shx', '.prj', '.cpg', '.qpj']:
                        aux_file = os.path.join(base_dir, f"{base_name}{ext}")
                        if os.path.exists(aux_file):
                            all_output_files.append(aux_file)
        
        detection_tasks[task_id]["status"] = "processing_complete"
        detection_tasks[task_id]["message"] = "模型处理完成，等待外部复制结果"
        detection_tasks[task_id]["result"] = { 
            **model_result_data,
            "docker_output_files": all_output_files
        }
        detection_tasks[task_id]["output_path"] = docker_internal_output_path
        
    except Exception as e:
        error_msg = f"任务失败: {str(e)}"
        if task_id in detection_tasks:
            detection_tasks[task_id]["status"] = "failed"
            detection_tasks[task_id]["message"] = error_msg
            if "error" not in model_result_data: model_result_data["error"] = str(e)
            detection_tasks[task_id]["result"] = model_result_data
        else:
            pass 
    finally:
        if task_id in detection_tasks and detection_tasks[task_id]["status"] not in ["pending", "running"]:
            detection_tasks[task_id]["end_time"] = datetime.now().isoformat()



# 获取任务状态接口
@app.get("/tasks/{task_id}", response_model=TaskStatusDetail)
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in detection_tasks:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    
    return detection_tasks[task_id]

# 获取所有任务列表
@app.get("/tasks", response_model=List[TaskStatusDetail])
async def list_tasks(limit: int = 10, status: Optional[str] = None):
    """获取任务列表"""
    tasks = list(detection_tasks.values())
    
    # 按状态过滤
    if status:
        tasks = [task for task in tasks if task["status"] == status]
    
    # 按开始时间降序排序
    tasks.sort(key=lambda x: x["start_time"], reverse=True)
    
    # 限制数量
    return tasks[:limit]




