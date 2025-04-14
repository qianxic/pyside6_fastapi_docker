import os
import requests
import json
import time
import shutil
import uuid
import subprocess  # Keep import in case needed elsewhere, though cp is removed
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 移除全局变量
# user_output_path_global: Optional[str] = None
DOCKER_CONTAINER_NAME = os.environ.get("DOCKER_CONTAINER_NAME", "change3d_api_container") # 容器名称

class PathConnector:
    """路径连接器类，用于从遥感系统中获取路径并通过API进行处理"""
    
    def __init__(self, api_url="http://localhost:8000", use_direct_copy=True):
        """初始化路径连接器"""
        self.api_url = api_url
        self.use_direct_copy = use_direct_copy
        self.task_contexts = {}
        self.process_endpoint = f"{api_url}/process/"
        self.tasks_endpoint = f"{api_url}/tasks/"
        if use_direct_copy:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.t1_dir = os.path.join(current_dir, "t1")
            self.t2_dir = os.path.join(current_dir, "t2")
            self.output_dir = os.path.join(current_dir, "output")
            self.docker_t1_dir = "/app/change3d_api_docker/t1"
            self.docker_t2_dir = "/app/change3d_api_docker/t2"
            self.docker_output_dir = "/app/change3d_api_docker/output"
            os.makedirs(self.t1_dir, exist_ok=True)
            os.makedirs(self.t2_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            
    def set_api_url(self, api_url):
        """设置API服务地址"""
        self.api_url = api_url
        self.process_endpoint = f"{api_url}/process/"
        self.tasks_endpoint = f"{api_url}/tasks/"
    
    def check_connection(self) -> bool:
        """检查API连接状态"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _generate_session_id(self) -> str:
        """生成唯一会话ID"""
        return f"{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _copy_to_t1(self, source_path: str, session_id: str) -> Tuple[str, str]:
        """复制前时相文件到t1目录 (无打印)"""
        filename = os.path.basename(source_path)
        temp_filename = f"{session_id}_{filename}"
        local_temp_path = os.path.join(self.t1_dir, temp_filename)
        docker_temp_path = f"{self.docker_t1_dir}/{temp_filename}"
        copy_success = False
        try:
            shutil.copy2(source_path, local_temp_path)
            if os.path.exists(local_temp_path):
                copy_success = True
        except Exception as e:
            # Propagate error
             raise IOError(f"复制前时相文件时出错: {source_path} -> {local_temp_path}, 错误: {str(e)}") from e
        if not copy_success:
            raise IOError(f"未能成功复制文件到宿主机共享目录: {local_temp_path}")
        return local_temp_path, docker_temp_path
    
    def _copy_directory_to_t1(self, source_dir: str, session_id: str) -> Tuple[str, str]:
        """复制前时相目录到t1目录 (无打印)"""
        dir_name = os.path.basename(source_dir)
        if not dir_name: dir_name = os.path.basename(os.path.dirname(source_dir))
        temp_dirname = f"{session_id}_{dir_name}"
        local_temp_dir = os.path.join(self.t1_dir, temp_dirname)
        docker_temp_dir = f"{self.docker_t1_dir}/{temp_dirname}"
        copy_success = False
        try:
            if os.path.exists(local_temp_dir):
                shutil.rmtree(local_temp_dir)
            shutil.copytree(source_dir, local_temp_dir)
            if os.path.isdir(local_temp_dir):
                copy_success = True
        except Exception as e:
             raise IOError(f"复制前时相目录时出错: {source_dir} -> {local_temp_dir}, 错误: {str(e)}") from e
        if not copy_success:
            raise IOError(f"未能成功复制目录到宿主机共享目录: {local_temp_dir}")
        return local_temp_dir, docker_temp_dir
    
    def _copy_to_t2(self, source_path: str, session_id: str) -> Tuple[str, str]:
        """复制后时相文件到t2目录 (无打印)"""
        filename = os.path.basename(source_path)
        temp_filename = f"{session_id}_{filename}"
        local_temp_path = os.path.join(self.t2_dir, temp_filename)
        docker_temp_path = f"{self.docker_t2_dir}/{temp_filename}"
        copy_success = False
        try:
            shutil.copy2(source_path, local_temp_path)
            if os.path.exists(local_temp_path):
                copy_success = True
        except Exception as e:
             raise IOError(f"复制后时相文件时出错: {source_path} -> {local_temp_path}, 错误: {str(e)}") from e
        if not copy_success:
            raise IOError(f"未能成功复制文件到宿主机共享目录: {local_temp_path}")
        return local_temp_path, docker_temp_path
    
    def _copy_directory_to_t2(self, source_dir: str, session_id: str) -> Tuple[str, str]:
        """复制后时相目录到t2目录 (无打印)"""
        dir_name = os.path.basename(source_dir)
        if not dir_name: dir_name = os.path.basename(os.path.dirname(source_dir))
        temp_dirname = f"{session_id}_{dir_name}"
        local_temp_dir = os.path.join(self.t2_dir, temp_dirname)
        docker_temp_dir = f"{self.docker_t2_dir}/{temp_dirname}"
        copy_success = False
        try:
            if os.path.exists(local_temp_dir):
                shutil.rmtree(local_temp_dir)
            shutil.copytree(source_dir, local_temp_dir)
            if os.path.isdir(local_temp_dir):
                copy_success = True
        except Exception as e:
             raise IOError(f"复制后时相目录时出错: {source_dir} -> {local_temp_dir}, 错误: {str(e)}") from e
        if not copy_success:
            raise IOError(f"未能成功复制目录到宿主机共享目录: {local_temp_dir}")
        return local_temp_dir, docker_temp_dir
    
    def _prepare_output_directory(self, output_dir_base: str, session_id: str) -> Tuple[str, str, str]:
        """准备输出目录 (无打印)"""
        # Use the basename of the intended *final* output dir for uniqueness
        dir_name = os.path.basename(output_dir_base) 
        if not dir_name: dir_name = f"output_{session_id}" # Fallback name
        temp_dirname = f"{session_id}_{dir_name}"
        local_temp_dir = os.path.join(self.output_dir, temp_dirname)
        docker_temp_dir = f"{self.docker_output_dir}/{temp_dirname}"
        os.makedirs(local_temp_dir, exist_ok=True)
        return local_temp_dir, docker_temp_dir, temp_dirname
    

    def detect_changes(self, before_path: str, after_path: str, 
                      output_path: str, mode: str = "single_image") -> Dict[str, Any]:
        """执行变化检测 (无打印)"""
        return self.process_with_direct_copy(before_path, after_path, output_path, mode)
    
    def send_request(self, endpoint: str, data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """发送请求到指定端点 (无打印)"""
        try:
            response = requests.post(endpoint, json=data, timeout=timeout)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": f"API请求失败: {str(e)}"}
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态 (无打印)"""
        try:
            response = requests.get(f"{self.tasks_endpoint}{task_id}", timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": f"获取任务状态失败: {str(e)}"}
    
    def list_tasks(self, limit: int = 10, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取任务列表 (无打印)"""
        try:
            url = f"{self.tasks_endpoint}?limit={limit}"
            if status: url += f"&status={status}"
            response = requests.get(url, timeout=10)
            return response.json()
        except Exception as e:
            return [{"status": "error", "message": f"获取任务列表失败: {str(e)}"}]



    def process_with_direct_copy(self, before_path: str, after_path: str, 
                               output_path: str, mode: str = "single_image") -> Dict[str, Any]:
        """使用直接文件复制的方式处理变化检测 (无打印)"""
        try:
            if not os.path.exists(before_path):
                return {"status": "error", "message": f"前时相路径不存在: {before_path}"}
            if not os.path.exists(after_path):
                return {"status": "error", "message": f"后时相路径不存在: {after_path}"}
            
            session_id = self._generate_session_id()
            original_output_path = output_path # Store user's intended final path
            
            if mode in ["single_image", "single_raster"]:
                _, docker_before_path = self._copy_to_t1(before_path, session_id)
                _, docker_after_path = self._copy_to_t2(after_path, session_id)
                _, docker_output_path, _ = self._prepare_output_directory(output_path, session_id)
            elif mode in ["batch_image", "batch_raster"]:
                _, docker_before_path = self._copy_directory_to_t1(before_path, session_id)
                _, docker_after_path = self._copy_directory_to_t2(after_path, session_id)
                _, docker_output_path, _ = self._prepare_output_directory(output_path, session_id)
            else:
                return {"status": "error", "message": f"不支持的处理模式: {mode}"}
            
            data = {
                "mode": mode,
                "before_path": docker_before_path,
                "after_path": docker_after_path,
                "output_path": docker_output_path
            }
            
            response = requests.post(f"{self.api_url}/detect/{mode}", json=data, timeout=30)
            
            if response.status_code != 200:
                return {"status": "error", "message": f"API请求失败: {response.status_code}", "detail": response.text}
            
            result = response.json()
            task_id = result.get("task_id")
            if not task_id:
                return {"status": "error", "message": "未获取到有效的任务ID"}
            
            # Store context needed for copying/renaming
            self.task_contexts[task_id] = {
                "session_id": session_id,
                "docker_output_path": docker_output_path, # Temp docker output dir
                "original_user_output_path": original_output_path, # User's final target dir
                "mode": mode # Store the mode for renaming logic
            }
            
            # Wait for completion using the polling mechanism
            task_result = self.wait_for_task_completion(task_id)

            # 打印接收到的状态和任务ID
            received_status = task_result.get('status')
            logging.info(f"### Received result from wait_for_task_completion for task {task_id}. Status: {received_status}")
            
            # 如果状态是 processing_complete，打印特定日志
            if received_status == "processing_complete":
                logging.info(f"###接收到状态id{{{received_status}}}")

            # --- 后处理 (复制和重命名) ---
            # 如果任务状态是 processing_complete，意味着模型已成功完成。
            # 我们现在处理复制/重命名，并确定最终状态 ('completed' 或 'copy_failed')。

            if task_result.get("status") == "processing_complete":
                logging.info(f"### Task {task_id}: 进入复制/重命名块，目前状态是 'processing_complete'。")
                context = self.task_contexts.get(task_id)

                # ---> 获取路径信息 <---
                original_user_output_path = context.get("original_user_output_path")
                docker_output_path = task_result.get("output_path") or context.get("docker_output_path")
                mode = context.get("mode")
                
                # !!! 检查点 1: 如果这里的 context 或路径获取失败 !!!
                if not original_user_output_path or not docker_output_path or not mode:
                    error_msg = f"任务 {task_id} 失败：未能获取复制/重命名所需的上下文信息。" # <--- 更具体的错误信息
                    logging.error(f"### Task {task_id}: {error_msg}")
                    task_result["status"] = "copy_failed"
                    task_result["message"] = error_msg
                    if task_id in self.task_contexts: del self.task_contexts[task_id]
                    return task_result # <--- 这里提前返回了 "copy_failed"

                # ---> 计算宿主机源目录 <--- 
                host_source_dir = None
                try:
                    if not isinstance(docker_output_path, str):
                        raise ValueError(f"无效的 docker_output_path 类型: {type(docker_output_path)}")
                    relative_output_subdir = os.path.basename(docker_output_path)
                    if not relative_output_subdir:
                         relative_output_subdir = os.path.basename(os.path.dirname(docker_output_path))
                    host_source_dir = os.path.join(self.output_dir, relative_output_subdir)
                    if not os.path.isdir(host_source_dir):
                        # ---> 如果源目录不存在，这是一个关键错误 <--- 
                        raise FileNotFoundError(f"模型输出的宿主机源目录未找到: {host_source_dir}")
                except Exception as path_e:
                    error_msg = f"计算用于复制的宿主机源路径时出错: {str(path_e)}"
                    logging.error(f"### Task {task_id}: 严重错误 - {error_msg}")
                    task_result["status"] = "copy_failed"
                    task_result["message"] = error_msg
                    if task_id in self.task_contexts: del self.task_contexts[task_id]
                    return task_result

                # --- 实际复制和重命名 (包裹在try/except中) ---
                display_image_file = None
                try: 
                    logging.info(f"### Task {task_id}: 尝试从 {host_source_dir} 复制到 {original_user_output_path}")
                    os.makedirs(original_user_output_path, exist_ok=True)
                    # 使用 copytree 并确保目标目录如果存在，则合并内容
                    shutil.copytree(host_source_dir, original_user_output_path, dirs_exist_ok=True)
                    display_image_file = self._rename_output_files(original_user_output_path, mode)
                    logging.info(f"### Task {task_id}: 复制和重命名成功完成。显示图像路径: {display_image_file}")

                    # --- 成功路径：设置最终状态 --- 
                    final_status = "completed"
                    final_message = f"任务完成。结果已复制并重命名到: {original_user_output_path}"
                    logging.info(f"### Task {task_id}: 设置最终状态为 '{final_status}'.")
                    task_result["status"] = final_status
                    task_result["message"] = final_message
                    task_result["output_path"] = original_user_output_path # 更新为最终用户路径
                    task_result["display_image_path"] = display_image_file

                except Exception as copy_rename_e: # <--- 捕获复制/重命名阶段的错误
                    error_msg = f"任务 {task_id} 失败：在复制或重命名输出文件时出错: {str(copy_rename_e)}"
                    logging.error(f"### Task {task_id}: {error_msg}", exc_info=True) # Log with traceback
                    task_result["status"] = "copy_failed" # 设置为 copy_failed
                    task_result["message"] = error_msg
                    # 尝试保留 output_path 和 display_image_path
                    task_result["output_path"] = original_user_output_path
                    task_result["display_image_path"] = display_image_file # 可能为 None
                    # 注意：这里不再提前返回，让函数流程继续到清理和最终返回

            else:
                # 如果状态不是 processing_complete (例如直接是 failed 或 completed)
                logging.info(f"### Task {task_id}: 状态是 '{task_result.get('status')}', 跳过复制/重命名块。")
                pass # 保持 task_result 不变

            # ---> 清理上下文 (总是在最后执行) <--- 
            logging.info(f"### Task {task_id}: 清理上下文。")
            if task_id in self.task_contexts:
                del self.task_contexts[task_id]

            # ---> 返回最终结果 <--- 
            logging.info(f"### Task {task_id}: 返回最终结果: Status='{task_result.get('status')}', Msg='{task_result.get('message')}'")
            return task_result # 返回最终更新后的 task_result
        except Exception as e:
            error_message = f"PathConnector 在 process_with_direct_copy 中处理错误: {str(e)}"
            logging.exception(f"### Task UNKNOWN (错误发生在早期): {error_message}") # 记录完整 traceback
            return {"status": "error", "message": error_message}
    
    

    
    def _rename_output_files(self, final_output_dir: str, mode: str) -> Optional[str]:
        """根据模式在最终输出目录中重命名文件。
        
        Args:
            final_output_dir: 用户的最终输出目录路径。
            mode: 处理模式 ('single_image', 'single_raster' 等)。
            
        Returns:
            主显示图像（重命名后的掩码）的完整路径，如果未找到则为 None。
        """
        display_image_path = None
        renamed_files_log = [] # 跟踪重命名操作
        vector_dir = os.path.join(final_output_dir, "vectors")

        try:
            if not os.path.isdir(final_output_dir):
                return None # 如果目录不存在则无需重命名

            items = os.listdir(final_output_dir)
            for item_name in items:
                item_path = os.path.join(final_output_dir, item_name)

                if os.path.isfile(item_path):
                    new_name = None
                    target_path = None
                    is_display_mask = False

                    # --- 图像模式重命名 ---
                    if mode in ["single_image", "batch_image"]:
                        if item_name.endswith("_result.png"):
                            new_name = "detection_mask.png"
                            is_display_mask = True
                        elif item_name.endswith("_quadview.png"):
                            new_name = "detection_visualization.png"

                    # --- 栅格模式重命名 ---
                    elif mode in ["single_raster", "batch_raster"]:
                        if item_name.endswith("_mask.tif"): 
                            new_name = "detection_mask.tif"
                            is_display_mask = True # This is the primary display mask
                        elif item_name.endswith("_result.tif"): 
                            new_name = "detection_result.tif" # Rename result.tif to detection_result.tif
                            # is_display_mask remains False for this file


                    # 如果确定了新名称，则执行重命名
                    if new_name:
                        target_path = os.path.join(final_output_dir, new_name)
                        try:
                            # 如果目标已存在（来自之前的文件），避免重命名
                            if not os.path.exists(target_path):
                                os.rename(item_path, target_path)
                                renamed_files_log.append(f"{item_name} -> {new_name}")
                                if is_display_mask:
                                    display_image_path = target_path
                            elif is_display_mask and display_image_path is None:
                                # 如果目标掩码已存在（可能已重命名），则捕获其路径
                                display_image_path = target_path
                        except OSError as rename_e:
                             # 记录或处理重命名错误，例如文件繁忙
                             pass 

                # --- 栅格矢量重命名 ---
                elif os.path.isdir(item_path) and item_name == "vectors" and mode in ["single_raster", "batch_raster"]:
                    vector_items = os.listdir(item_path)
                    for vec_item_name in vector_items:
                        vec_item_path = os.path.join(item_path, vec_item_name)
                        if os.path.isfile(vec_item_path):
                            vec_new_name = None
                            vec_target_path = None
                            _, ext = os.path.splitext(vec_item_name)
                            ext = ext.lower()

                            if ext == '.shp': vec_new_name = "changes.shp"
                            elif ext == '.geojson': vec_new_name = "changes.geojson"
                            elif ext == '.dbf': vec_new_name = "changes.dbf"
                            elif ext == '.shx': vec_new_name = "changes.shx"
                            elif ext == '.prj': vec_new_name = "changes.prj"
                            elif ext == '.cpg': vec_new_name = "changes.cpg"
                            elif ext == '.qpj': vec_new_name = "changes.qpj"

                            if vec_new_name:
                                vec_target_path = os.path.join(item_path, vec_new_name)
                                try:
                                    if not os.path.exists(vec_target_path):
                                        os.rename(vec_item_path, vec_target_path)
                                        renamed_files_log.append(f"vectors/{vec_item_name} -> vectors/{vec_new_name}")
                                except OSError as vec_rename_e:
                                    # 记录或处理重命名错误
                                    pass 
            return display_image_path
        except Exception as e:
            # 记录重命名过程中的错误
            return display_image_path # 返回在错误之前找到的任何路径



    def wait_for_task_completion(self, task_id, poll_interval=15, short_timeout=10, max_wait_time=3600):
        """
        Waits for the task to complete model processing or fail.
        Returns the result when status is 'processing_complete', 'completed', 'failed', or 'copy_failed'.
        'processing_complete' signals that model processing is done and copy should start.
        'completed'/'failed'/'copy_failed' are terminal states from the API.
        """
        status_url = f"{self.api_url}/tasks/{task_id}"
        logging.info(f"Waiting for task {task_id} state (processing_complete/completed/failed/copy_failed). Polling: {status_url}")
        start_time = time.time()
    
        while True:
            current_time = time.time()
            if current_time - start_time > max_wait_time:
                logging.error(f"Task {task_id}: Polling timed out after {max_wait_time} seconds waiting for processing_complete/completed/failed.")
                return {"status": "error", "message": f"Polling timed out after {max_wait_time} seconds"}

            try:
                response = requests.get(status_url, timeout=short_timeout)
                response.raise_for_status()
                result = response.json()
                status = result.get("status")

                # Check for terminal states OR processing_complete (signal to start copy)
                if status in ["completed", "failed", "copy_failed", "processing_complete"]:
                    logging.info(f"Task {task_id} reached decisive status: {status}. Returning result.")
                    return result  # Exit loop and return the result
                elif status in ["pending", "running"]:
                    # Task still initializing or running model, continue polling
                    pass # Continue loop after sleep
                else:
                    # Unexpected status from API
                    logging.error(f"Task {task_id}: Received unexpected status '{status}'. Result: {result}")
                    return {"status": "error", "message": f"Unexpected status '{status}' received"}

            except requests.exceptions.Timeout:
                logging.warning(f"Task {task_id}: Status check timed out (timeout={short_timeout}s). Retrying after {poll_interval}s...")
            except requests.exceptions.RequestException as e:
                logging.error(f"Task {task_id}: Error checking status: {e}")
                logging.warning(f"Task {task_id}: Connection error during status check. Retrying after {poll_interval}s...")

            # Wait before the next poll
            time.sleep(poll_interval)

    def _handle_api_response(self, response):
        """Handles the API response and initiates waiting if task ID is received."""



class LazyPathConnector:
    """延迟初始化的PathConnector包装器"""
    def __init__(self):
        self._instance = None
    def __getattr__(self, name):
        if self._instance is None:
            self._instance = PathConnector()
        return getattr(self._instance, name)

path_connector = LazyPathConnector() 