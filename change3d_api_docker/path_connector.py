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
    """路径连接器类，用于与 API 服务进行交互 (简化版，移除了客户端文件处理逻辑)"""
    
    def __init__(self, api_url="http://localhost:8000", use_direct_copy=True):
        """初始化路径连接器 (简化版)"""
        self.api_url = api_url
        # self.use_direct_copy = use_direct_copy # No longer relevant here
        # self.task_contexts = {} # Removed, handled by client
        self.process_endpoint = f"{api_url}/process/" # Kept for potential future use or other methods
        self.tasks_endpoint = f"{api_url}/tasks/"
        # Removed shared directory setup logic
            
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
    
    # Removed _generate_session_id as it was mainly for temp file naming
    # Removed _copy_to_t1
    # Removed _copy_directory_to_t1
    # Removed _copy_to_t2
    # Removed _copy_directory_to_t2
    # Removed _prepare_output_directory
    # Removed detect_changes (as it called the removed process_with_direct_copy)
    
    def send_request(self, endpoint: str, data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """发送请求到指定端点 (无打印)"""
        try:
            response = requests.post(endpoint, json=data, timeout=timeout)
            response.raise_for_status() # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
             logging.error(f"API request to {endpoint} failed: {e}")
             # Provide more context in the error message
             error_detail = "Unknown error" 
             try:
                 # Try to get more details from the response if available
                 error_detail = e.response.text if e.response is not None else str(e)
             except: pass
             return {"status": "error", "message": f"API请求失败: {str(e)}", "detail": error_detail}
        except Exception as e:
             # Catch other potential errors (like JSON decoding)
             logging.error(f"Error during API request to {endpoint}: {e}")
             return {"status": "error", "message": f"处理 API 请求时出错: {str(e)}"}
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态 (无打印)"""
        try:
            status_url = f"{self.tasks_endpoint}{task_id}"
            response = requests.get(status_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to get task status for {task_id}: {e}")
            error_detail = "Unknown error"
            try:
                error_detail = e.response.text if e.response is not None else str(e)
            except: pass
            # Consistent error format
            return {"status": "error", "message": f"获取任务 {task_id} 状态失败: {str(e)}", "detail": error_detail}
        except Exception as e:
            logging.error(f"Error getting task status for {task_id}: {e}")
            return {"status": "error", "message": f"处理获取任务状态时出错: {str(e)}"}

    
    def list_tasks(self, limit: int = 10, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取任务列表 (无打印)"""
        try:
            url = f"{self.tasks_endpoint}?limit={limit}"
            if status: url += f"&status={status}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to list tasks: {e}")
            error_detail = "Unknown error"
            try:
                 error_detail = e.response.text if e.response is not None else str(e)
            except: pass
            return [{"status": "error", "message": f"获取任务列表失败: {str(e)}", "detail": error_detail}]
        except Exception as e:
             logging.error(f"Error listing tasks: {e}")
             return [{"status": "error", "message": f"处理获取任务列表时出错: {str(e)}"}]

    # Removed process_with_direct_copy
    # Removed _rename_output_files
    # Removed wait_for_task_completion (now handled by client)
    # Removed _handle_api_response (was tied to removed methods)

class LazyPathConnector:
    """延迟初始化的PathConnector包装器"""
    def __init__(self):
        self._instance = None
    def __getattr__(self, name):
        if self._instance is None:
            # Pass use_direct_copy=False explicitly, although it's not used in the simplified init
            self._instance = PathConnector(use_direct_copy=False)
        return getattr(self._instance, name)

path_connector = LazyPathConnector() 