import requests
import json
import os
import shutil
import uuid
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# 配置客户端的基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置 ---
# API基础URL，可通过配置文件或环境变量设置
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")  # 修改为正确的端口8000

# 客户端共享卷路径 (使用相对路径，避免硬编码绝对路径)
# 重要提示: 这些路径必须与挂载到 Docker 容器中的主机路径相对应
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T1_DIR_CLIENT = os.path.join(project_root, "change3d_api_docker", "t1")
T2_DIR_CLIENT = os.path.join(project_root, "change3d_api_docker", "t2")
OUTPUT_DIR_CLIENT = os.path.join(project_root, "change3d_api_docker", "output")

# Docker 内部路径 (必须与容器内的目标路径匹配)
T1_DIR_DOCKER = "/app/change3d_api_docker/t1"
T2_DIR_DOCKER = "/app/change3d_api_docker/t2"
OUTPUT_DIR_DOCKER = "/app/change3d_api_docker/output"

# 确保客户端共享目录存在
os.makedirs(T1_DIR_CLIENT, exist_ok=True)
os.makedirs(T2_DIR_CLIENT, exist_ok=True)
os.makedirs(OUTPUT_DIR_CLIENT, exist_ok=True)

# --- 辅助函数 (改编自 path_connector.py) ---

def _generate_session_id() -> str:
    """生成唯一会话ID"""
    return f"{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

def _copy_to_shared(source_path: str, target_dir_client: str, target_dir_docker: str, session_id: str) -> Tuple[str, str]:
    """将文件复制到客户端共享目录，并返回客户端和Docker路径。"""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"源文件未找到: {source_path}")
    
    # 确保目标目录存在
    os.makedirs(target_dir_client, exist_ok=True)
    
    filename = os.path.basename(source_path)
    temp_filename = f"{session_id}_{filename}"
    local_temp_path = os.path.join(target_dir_client, temp_filename)
    docker_temp_path = f"{target_dir_docker}/{temp_filename}"
    
    logging.info(f"开始复制文件: {source_path} -> {local_temp_path}")
    logging.info(f"Docker路径将是: {docker_temp_path}")
    
    try:
        shutil.copy2(source_path, local_temp_path)
        if not os.path.exists(local_temp_path):
             raise IOError(f"未能将文件复制到共享目录: {local_temp_path}")
        
        # 验证文件大小
        source_size = os.path.getsize(source_path)
        target_size = os.path.getsize(local_temp_path)
        logging.info(f"文件复制完成: {source_path} ({source_size} bytes) -> {local_temp_path} ({target_size} bytes)")
        
        if source_size != target_size:
            logging.warning(f"文件大小不匹配: 源={source_size}, 目标={target_size}")
        
        return local_temp_path, docker_temp_path
    except Exception as e:
        logging.error(f"复制文件失败: {source_path} -> {local_temp_path}, 错误: {str(e)}")
        raise IOError(f"复制文件 {source_path} 到 {local_temp_path} 时出错: {str(e)}") from e

def _copy_directory_to_shared(source_dir: str, target_dir_client: str, target_dir_docker: str, session_id: str) -> Tuple[str, str]:
    """将目录复制到客户端共享目录，并返回客户端和Docker路径。"""
    if not os.path.isdir(source_dir):
         raise NotADirectoryError(f"源路径不是目录: {source_dir}")
    dir_name = os.path.basename(source_dir.rstrip('/\\'))
    if not dir_name: dir_name = os.path.basename(os.path.dirname(source_dir.rstrip('/\\')))
    temp_dirname = f"{session_id}_{dir_name}"
    local_temp_dir = os.path.join(target_dir_client, temp_dirname)
    docker_temp_dir = f"{target_dir_docker}/{temp_dirname}"
    try:
        if os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir) # 确保状态干净
        shutil.copytree(source_dir, local_temp_dir)
        if not os.path.isdir(local_temp_dir):
             raise IOError(f"未能将目录复制到共享目录: {local_temp_dir}")
        logging.info(f"已复制目录 {source_dir} 到 {local_temp_dir}")
        return local_temp_dir, docker_temp_dir
    except Exception as e:
        raise IOError(f"复制目录 {source_dir} 到 {local_temp_dir} 时出错: {str(e)}") from e

def _prepare_output_directory(output_dir_client: str, output_dir_docker: str, user_output_path_base: str, session_id: str) -> Tuple[str, str]:
    """准备客户端共享输出目录，并返回客户端和Docker路径。"""
    # 使用预期的 *最终* 用户输出目录的基名以确保在共享空间中的唯一性
    dir_name = os.path.basename(user_output_path_base.rstrip('/\\'))
    if not dir_name: dir_name = f"output_{session_id}" # 后备名称
    temp_dirname = f"{session_id}_{dir_name}"
    local_temp_dir = os.path.join(output_dir_client, temp_dirname)
    docker_temp_dir = f"{output_dir_docker}/{temp_dirname}"
    try:
        os.makedirs(local_temp_dir, exist_ok=True) # 创建特定的会话输出目录
        logging.info(f"已准备客户端输出目录: {local_temp_dir}")
        return local_temp_dir, docker_temp_dir
    except Exception as e:
        raise IOError(f"准备输出目录 {local_temp_dir} 时出错: {str(e)}") from e

def _wait_for_task_completion(task_id, poll_interval=15, short_timeout=10, max_wait_time=3600) -> Dict[str, Any]:
    """轮询 API 获取任务状态，直到达到最终状态。"""
    status_url = f"{API_BASE_URL}/tasks/{task_id}"
    logging.info(f"等待任务 {task_id} 完成。轮询地址: {status_url}")
    start_time = time.time()

    while True:
        current_time = time.time()
        if current_time - start_time > max_wait_time:
            logging.error(f"任务 {task_id}: 轮询在 {max_wait_time} 秒后超时。")
            return {"status": "error", "message": f"轮询在 {max_wait_time} 秒后超时"}

        try:
            response = requests.get(status_url, timeout=short_timeout)
            response.raise_for_status() # 对错误的响应 (4xx 或 5xx) 抛出 HTTPError
            result = response.json()
            status = result.get("status")

            # 检查终止状态或 processing_complete (表示开始复制)
            if status in ["completed", "failed", "copy_failed", "processing_complete"]:
                logging.info(f"任务 {task_id} 达到决定性状态: {status}。返回 API 结果。")
                return result # 退出循环并返回 API 的结果
            elif status in ["pending", "running"]:
                logging.debug(f"任务 {task_id} 状态: {status}。继续轮询...")
                pass # 睡眠后继续循环
            else:
                logging.error(f"任务 {task_id}: 收到意外状态 '{status}'。API 结果: {result}")
                return {"status": "error", "message": f"从 API 收到意外状态 '{status}'"}

        except requests.exceptions.Timeout:
            logging.warning(f"任务 {task_id}: 状态检查超时 (超时时间={short_timeout}秒)。将在 {poll_interval} 秒后重试...")
        except requests.exceptions.RequestException as e:
            logging.error(f"任务 {task_id}: 检查状态时出错: {e}")
            # 决定是在连接错误时重试还是立即失败
            # 目前，延迟后重试
            logging.warning(f"任务 {task_id}: 状态检查期间连接错误。将在 {poll_interval} 秒后重试...")

        time.sleep(poll_interval)

def _rename_output_files(final_output_dir: str, mode: str, session_id: str = None) -> Optional[str]:
    """
    在最终用户输出目录中查找、重命名文件，并返回主显示图像的预期路径。
    Args:
        final_output_dir: 用户的最终本地输出目录路径。
        mode: 处理模式 ('single_image', 'single_raster' 等)。
        session_id: 当前任务的会话ID，用于筛选正确的文件。
    Returns:
        主显示图像（掩码）的预期完整路径 (例如 '.../detection_mask.png')，
        如果找不到源掩码文件则为 None。
    """
    expected_display_path = None # 存储预期的最终显示路径
    source_mask_path = None     # 存储找到的原始掩码文件路径
    source_vis_path = None      # 存储找到的原始可视化文件路径
    source_result_tif_path = None # 存储找到的原始栅格结果文件路径

    # 重命名后的文件名包含session_id，便于后续根据任务ID选择文件
    if session_id:
        target_mask_name = f"{session_id}_detection_mask.png" if mode in ["single_image", "batch_image"] else f"{session_id}_detection_mask.tif"
        target_vis_name = f"{session_id}_detection_visualization.png"
        target_result_tif_name = f"{session_id}_detection_result.tif"
    else:
        target_mask_name = "detection_mask.png" if mode in ["single_image", "batch_image"] else "detection_mask.tif"
        target_vis_name = "detection_visualization.png"
        target_result_tif_name = "detection_result.tif"

    logging.info(f"在目录 {final_output_dir} 中开始重命名过程，模式: {mode}, 会话ID: {session_id}")

    try:
        if not os.path.isdir(final_output_dir):
            logging.warning(f"最终输出目录不存在: {final_output_dir}")
            return None

        items = os.listdir(final_output_dir)
        logging.info(f"在 {final_output_dir} 中找到的项目: {items}")
        
        # 筛选出属于当前任务的文件
        if session_id:
            session_items = [item for item in items if session_id in item]
            logging.info(f"属于当前任务 {session_id} 的文件: {session_items}")
        else:
            logging.warning("未提供session_id，将处理所有匹配的文件")
            session_items = items

        # --- 第一步: 识别源文件 ---
        for item_name in session_items:
            item_path = os.path.join(final_output_dir, item_name)
            if not os.path.isfile(item_path):
                continue

            item_name_lower = item_name.lower()

            # 识别潜在的源掩码文件
            if mode in ["single_image", "batch_image"] and item_name_lower.endswith("_result.png"):
                source_mask_path = item_path
                logging.debug(f"识别到潜在的图像掩码源: {source_mask_path}")
            elif mode in ["single_raster", "batch_raster"] and item_name_lower.endswith("_mask.tif"):
                source_mask_path = item_path
                logging.debug(f"识别到潜在的栅格掩码源: {source_mask_path}")

            # 识别潜在的源可视化文件
            elif "quadview" in item_name_lower and item_name_lower.endswith(".png"):
                 source_vis_path = item_path
                 logging.debug(f"识别到潜在的可视化源: {source_vis_path}")

            # 识别潜在的源次要栅格结果文件
            elif mode in ["single_raster", "batch_raster"] and item_name_lower.endswith("_result.tif"):
                 source_result_tif_path = item_path
                 logging.debug(f"识别到潜在的次要栅格结果源: {source_result_tif_path}")

        # --- 第二步: 确定预期的显示路径并尝试重命名 ---
        rename_ops = []
        if source_mask_path:
            # 验证找到的文件确实属于当前任务
            if session_id and session_id not in os.path.basename(source_mask_path):
                logging.error(f"找到的掩码文件不属于当前任务: {source_mask_path}, 当前任务ID: {session_id}")
                return None
                
            expected_display_path = os.path.join(final_output_dir, target_mask_name)
            rename_ops.append((source_mask_path, expected_display_path))
            logging.info(f"找到当前任务的掩码文件: {source_mask_path} -> {expected_display_path}")
        else:
             logging.warning(f"在 {final_output_dir} 中找不到属于当前任务 {session_id} 的源掩码文件 (_result.png 或 _mask.tif)")

        if source_vis_path:
            target_vis_path = os.path.join(final_output_dir, target_vis_name)
            rename_ops.append((source_vis_path, target_vis_path))

        if source_result_tif_path:
            target_result_tif_path = os.path.join(final_output_dir, target_result_tif_name)
            rename_ops.append((source_result_tif_path, target_result_tif_path))

        renamed_files_log = []
        for source, target in rename_ops:
            # 如果源和目标因某种原因相同，则跳过
            if os.path.exists(source) and os.path.normcase(source) == os.path.normcase(target):
                continue

            try:
                # 检查目标是否存在且不是同一个文件
                if os.path.exists(target):
                     logging.warning(f"目标文件 {target} 已存在。跳过对 {source} 的重命名。")
                elif os.path.exists(source):
                    os.rename(source, target)
                    renamed_files_log.append(f"{os.path.basename(source)} -> {os.path.basename(target)}")
                    logging.info(f"成功将 {os.path.basename(source)} 重命名为 {os.path.basename(target)}")
                else:
                     logging.warning(f"未找到用于重命名的源文件 {source}。")
            except OSError as e:
                logging.warning(f"无法将 {os.path.basename(source)} 重命名为 {os.path.basename(target)}: {e}")
            except Exception as e:
                 logging.error(f"重命名 {os.path.basename(source)} 时发生意外错误: {e}")


        if renamed_files_log:
            logging.info(f"在 {final_output_dir} 中完成重命名操作。重命名的文件: {', '.join(renamed_files_log)}")
        else:
             logging.info(f"在 {final_output_dir} 中没有文件被重命名。")

        # 返回 *预期* 的路径，即使重命名失败 (例如，文件已存在)
        logging.info(f"重命名过程完成。返回预期的显示路径: {expected_display_path}")
        return expected_display_path

    except Exception as e:
        logging.error(f"在 {final_output_dir} 中进行文件重命名过程时出错: {e}", exc_info=True)
        return None # 如果过程严重失败，则返回 None

def find_result_file_by_session_id(output_dir: str, mode: str, session_id: str) -> Optional[str]:
    """
    根据任务ID查找重命名后的结果文件
    
    Args:
        output_dir: 输出目录路径
        mode: 处理模式 ('single_image', 'single_raster' 等)
        session_id: 任务ID
        
    Returns:
        结果文件的完整路径，如果找不到则返回None
    """
    if not session_id:
        logging.warning("未提供session_id，无法查找结果文件")
        return None
        
    if not os.path.isdir(output_dir):
        logging.warning(f"输出目录不存在: {output_dir}")
        return None
    
    # 根据模式确定要查找的文件名
    if mode in ["single_image", "batch_image"]:
        target_filename = f"{session_id}_detection_mask.png"
    elif mode in ["single_raster", "batch_raster"]:
        target_filename = f"{session_id}_detection_mask.tif"
    else:
        logging.error(f"不支持的模式: {mode}")
        return None
    
    target_path = os.path.join(output_dir, target_filename)
    
    if os.path.exists(target_path):
        logging.info(f"找到任务 {session_id} 的结果文件: {target_path}")
        return target_path
    else:
        logging.warning(f"未找到任务 {session_id} 的结果文件: {target_path}")
        return None

def cleanup_old_result_files(output_dir: str, current_session_id: str, mode: str):
    """
    清理旧的重命名结果文件，只保留当前任务的文件
    
    Args:
        output_dir: 输出目录路径
        current_session_id: 当前任务的session_id
        mode: 处理模式
    """
    if not os.path.isdir(output_dir):
        return
        
    try:
        # 查找所有重命名的结果文件
        pattern = "*_detection_mask.*" if mode in ["single_image", "batch_image"] else "*_detection_mask.*"
        import glob
        
        for file_path in glob.glob(os.path.join(output_dir, pattern)):
            filename = os.path.basename(file_path)
            
            # 如果文件不属于当前任务，则删除
            if current_session_id not in filename:
                try:
                    os.remove(file_path)
                    logging.info(f"清理旧的结果文件: {filename}")
                except Exception as e:
                    logging.warning(f"删除旧文件失败: {filename}, 错误: {e}")
                    
    except Exception as e:
        logging.warning(f"清理旧结果文件时出错: {e}")

# --- 主要客户端函数 ---

def detect_changes(before_path: str, after_path: str,
                  output_path: str, mode: str = "single_image") -> Dict[str, Any]:
    """
    从客户端协调变化检测过程。
    处理文件复制、API 交互、轮询和结果检索。

    Args:
        before_path: 前时相图像/目录的本地路径。
        after_path: 后时相图像/目录的本地路径。
        output_path: 最终输出目录的本地路径。
        mode: 处理模式 ('single_image', 'single_raster', 'batch_image', 'batch_raster')。

    Returns:
        Dict: 最终结果状态和信息。
    """
    session_id = _generate_session_id()
    client_temp_output_dir = None # 用于存储客户端上的临时输出目录路径

    try:
        # --- 1. 准备输入并复制到共享卷 ---
        logging.info(f"[{session_id}] 开始检测 (模式: {mode})。之前: {before_path}, 之后: {after_path}, 输出: {output_path}")
        docker_before_path = ""
        docker_after_path = ""
        docker_output_path = ""

        if mode in ["single_image", "single_raster"]:
            if not os.path.isfile(before_path): return {"status": "error", "message": f"之前的路径不是文件: {before_path}"}
            if not os.path.isfile(after_path): return {"status": "error", "message": f"之后的路径不是文件: {after_path}"}
            _, docker_before_path = _copy_to_shared(before_path, T1_DIR_CLIENT, T1_DIR_DOCKER, session_id)
            _, docker_after_path = _copy_to_shared(after_path, T2_DIR_CLIENT, T2_DIR_DOCKER, session_id)
            client_temp_output_dir, docker_output_path = _prepare_output_directory(OUTPUT_DIR_CLIENT, OUTPUT_DIR_DOCKER, output_path, session_id)
        elif mode in ["batch_image", "batch_raster"]:
            if not os.path.isdir(before_path): return {"status": "error", "message": f"之前的路径不是目录: {before_path}"}
            if not os.path.isdir(after_path): return {"status": "error", "message": f"之后的路径不是目录: {after_path}"}
            _, docker_before_path = _copy_directory_to_shared(before_path, T1_DIR_CLIENT, T1_DIR_DOCKER, session_id)
            _, docker_after_path = _copy_directory_to_shared(after_path, T2_DIR_CLIENT, T2_DIR_DOCKER, session_id)
            client_temp_output_dir, docker_output_path = _prepare_output_directory(OUTPUT_DIR_CLIENT, OUTPUT_DIR_DOCKER, output_path, session_id)
        else:
            return {"status": "error", "message": f"不支持的模式: {mode}"}

        # --- 2. 调用 API 启动检测任务 ---
        api_endpoint = f"{API_BASE_URL}/detect/{mode}"
        # 修改payload结构，将参数放在根级别
        payload = {
            "mode": mode,
            "before_path": docker_before_path,
            "after_path": docker_after_path,
            "output_path": docker_output_path
        }
        
        logging.info(f"[{session_id}] 发送请求到 API: {api_endpoint}，载荷: {payload}")
        headers = {"Content-Type": "application/json"}  # 添加内容类型头
        response = requests.post(api_endpoint, json=payload, headers=headers, timeout=30)
        
        # 添加详细的错误日志
        if response.status_code == 422:
            error_detail = response.json()
            logging.error(f"[{session_id}] 请求参数验证失败: {error_detail}")
            return {"status": "error", "message": f"API请求参数验证失败，请检查参数格式"}

        response.raise_for_status() # 检查 HTTP 错误

        api_start_result = response.json()
        task_id = api_start_result.get("task_id")
        if not task_id:
            logging.error(f"[{session_id}] API 未返回有效的 task_id。响应: {api_start_result}")
            return {"status": "error", "message": "API 未返回有效的 task_id", "detail": api_start_result}
        logging.info(f"[{session_id}] 任务创建成功。任务 ID: {task_id}")

        # --- 3. 等待任务完成 ---
        task_result_api = _wait_for_task_completion(task_id)
        api_status = task_result_api.get("status")
        api_message = task_result_api.get("message", "")

        # 处理通过轮询报告的 API 端故障
        if api_status not in ["processing_complete", "completed"]: # 如果 API 处理复制，则可能返回 completed，但我们的客户端期望 processing_complete
            logging.error(f"[{session_id}] 任务 {task_id} 在 API 端失败或意外结束。状态: {api_status}, 消息: {api_message}")
            # 返回 API 报告的错误
            return task_result_api # 这个字典已经包含 status='error' 或 'failed' 等

        # --- 4. 处理结果 (从共享卷复制并重命名) ---
        logging.info(f"[{session_id}] 任务 {task_id} 处理完成 (API 状态: {api_status})。开始结果检索和重命名。")

        if not client_temp_output_dir or not os.path.isdir(client_temp_output_dir):
             logging.error(f"[{session_id}] 客户端临时输出目录未找到或无效: {client_temp_output_dir}")
             return {"status": "copy_failed", "message": f"客户端临时输出目录未找到: {client_temp_output_dir}", "task_id": task_id}

        final_display_image_path = None
        try:
            # 确保最终用户输出目录存在
            os.makedirs(output_path, exist_ok=True)
            # 将结果从临时共享客户端目录复制到最终用户目录
            logging.info(f"[{session_id}] 从 {client_temp_output_dir} 复制结果到 {output_path}")
            # 使用 copytree 和 dirs_exist_ok=True 在最终目录已存在时合并内容
            shutil.copytree(client_temp_output_dir, output_path, dirs_exist_ok=True)

            # 在最终用户目录中重命名文件
            logging.info(f"[{session_id}] 在 {output_path} 中重命名输出文件")
            _rename_output_files(output_path, mode, session_id)

            # 根据任务ID查找重命名后的结果文件
            final_display_image_path = find_result_file_by_session_id(output_path, mode, session_id)
            
            # 清理旧的重命名结果文件
            cleanup_old_result_files(output_path, session_id, mode)

            # --- 5. 构建最终成功结果 ---
            final_status = "completed"
            final_message = f"任务成功完成。结果已保存到: {output_path}"
            logging.info(f"[{session_id}] {final_message}")
            return {
                "status": final_status,
                "message": final_message,
                "task_id": task_id,
                "session_id": session_id,  # 添加session_id用于调试和验证
                "output_path": output_path, # 最终用户路径
                "display_image_path": final_display_image_path # 用户目录中重命名后的路径
                # 如果需要，可以选择性地包含原始 API 结果详情
                # "api_result": task_result_api.get("result", {})
            }

        except Exception as copy_rename_e:
            error_msg = f"结果复制或重命名期间失败: {str(copy_rename_e)}"
            logging.error(f"[{session_id}] {error_msg}", exc_info=True)
            return {
                "status": "copy_failed", # 客户端复制/重命名失败的特定状态
                "message": error_msg,
                "task_id": task_id,
                "output_path": output_path, # 仍然报告预期的输出路径
                "display_image_path": final_display_image_path # 可能为 None 或部分重命名后的路径
            }
        # --- 清理临时客户端输出目录 ---
        # finally:
        #     if client_temp_output_dir and os.path.isdir(client_temp_output_dir):
        #         try:
        #             logging.info(f"[{session_id}] 清理临时输出目录: {client_temp_output_dir}")
        #             shutil.rmtree(client_temp_output_dir)
        #         except Exception as cleanup_e:
        #             logging.warning(f"[{session_id}] 清理临时目录 {client_temp_output_dir} 失败: {cleanup_e}")


    except Exception as e:
        error_message = f"在 detect_changes 中发生错误: {str(e)}"
        logging.error(f"[{session_id}] {error_message}", exc_info=True)
        return {"status": "error", "message": error_message}


# --- 连接和配置函数 ---

def check_connection() -> bool:
    """检查API连接状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.warning(f"API 连接检查失败: {e}")
        return False

def set_api_base_url(url: str):
    """设置API基础URL"""
    global API_BASE_URL
    API_BASE_URL = url
    logging.info(f"API 基础 URL 设置为: {url}")

def get_api_base_url() -> str:
    """获取当前API基础URL"""
    return API_BASE_URL
