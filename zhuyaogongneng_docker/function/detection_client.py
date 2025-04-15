import requests
import json
import os
import shutil
import uuid
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# Configure basic logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# API基础URL，可通过配置文件或环境变量设置
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Client-side paths for shared volumes (using provided absolute paths)
# IMPORTANT: These paths MUST correspond to the host paths mounted into the Docker container
T1_DIR_CLIENT = r"D:\代码\RSIIS\遥感影像变化检测系统V1.1\change3d_api_docker\t1" # Use raw string for Windows path
T2_DIR_CLIENT = r"D:\代码\RSIIS\遥感影像变化检测系统V1.1\change3d_api_docker\t2" # Use raw string for Windows path
OUTPUT_DIR_CLIENT = r"D:\代码\RSIIS\遥感影像变化检测系统V1.1\change3d_api_docker\output" # Use raw string for Windows path

# Docker internal paths (MUST match the target paths inside the container)
T1_DIR_DOCKER = "/app/change3d_api_docker/t1"
T2_DIR_DOCKER = "/app/change3d_api_docker/t2"
OUTPUT_DIR_DOCKER = "/app/change3d_api_docker/output"

# Ensure client-side shared directories exist
os.makedirs(T1_DIR_CLIENT, exist_ok=True)
os.makedirs(T2_DIR_CLIENT, exist_ok=True)
os.makedirs(OUTPUT_DIR_CLIENT, exist_ok=True)

# --- Helper Functions (Adapted from path_connector.py) ---

def _generate_session_id() -> str:
    """生成唯一会话ID"""
    return f"{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

def _copy_to_shared(source_path: str, target_dir_client: str, target_dir_docker: str, session_id: str) -> Tuple[str, str]:
    """Copies a file to the client-side shared directory and returns client/docker paths."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    filename = os.path.basename(source_path)
    temp_filename = f"{session_id}_{filename}"
    local_temp_path = os.path.join(target_dir_client, temp_filename)
    docker_temp_path = f"{target_dir_docker}/{temp_filename}"
    try:
        shutil.copy2(source_path, local_temp_path)
        if not os.path.exists(local_temp_path):
             raise IOError(f"Failed to copy file to shared directory: {local_temp_path}")
        logging.info(f"Copied {source_path} to {local_temp_path}")
        return local_temp_path, docker_temp_path
    except Exception as e:
        raise IOError(f"Error copying file {source_path} to {local_temp_path}: {str(e)}") from e

def _copy_directory_to_shared(source_dir: str, target_dir_client: str, target_dir_docker: str, session_id: str) -> Tuple[str, str]:
    """Copies a directory to the client-side shared directory and returns client/docker paths."""
    if not os.path.isdir(source_dir):
         raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
    dir_name = os.path.basename(source_dir.rstrip('/\\'))
    if not dir_name: dir_name = os.path.basename(os.path.dirname(source_dir.rstrip('/\\')))
    temp_dirname = f"{session_id}_{dir_name}"
    local_temp_dir = os.path.join(target_dir_client, temp_dirname)
    docker_temp_dir = f"{target_dir_docker}/{temp_dirname}"
    try:
        if os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir) # Ensure clean state
        shutil.copytree(source_dir, local_temp_dir)
        if not os.path.isdir(local_temp_dir):
             raise IOError(f"Failed to copy directory to shared directory: {local_temp_dir}")
        logging.info(f"Copied directory {source_dir} to {local_temp_dir}")
        return local_temp_dir, docker_temp_dir
    except Exception as e:
        raise IOError(f"Error copying directory {source_dir} to {local_temp_dir}: {str(e)}") from e

def _prepare_output_directory(output_dir_client: str, output_dir_docker: str, user_output_path_base: str, session_id: str) -> Tuple[str, str]:
    """Prepares the client-side shared output directory and returns client/docker paths."""
    # Use the basename of the intended *final* user output dir for uniqueness in the shared space
    dir_name = os.path.basename(user_output_path_base.rstrip('/\\'))
    if not dir_name: dir_name = f"output_{session_id}" # Fallback name
    temp_dirname = f"{session_id}_{dir_name}"
    local_temp_dir = os.path.join(output_dir_client, temp_dirname)
    docker_temp_dir = f"{output_dir_docker}/{temp_dirname}"
    try:
        os.makedirs(local_temp_dir, exist_ok=True) # Create the specific session output dir
        logging.info(f"Prepared client output directory: {local_temp_dir}")
        return local_temp_dir, docker_temp_dir
    except Exception as e:
        raise IOError(f"Error preparing output directory {local_temp_dir}: {str(e)}") from e

def _wait_for_task_completion(task_id, poll_interval=15, short_timeout=10, max_wait_time=3600) -> Dict[str, Any]:
    """Polls the API for task status until a final state is reached."""
    status_url = f"{API_BASE_URL}/tasks/{task_id}"
    logging.info(f"Waiting for task {task_id} completion. Polling: {status_url}")
    start_time = time.time()

    while True:
        current_time = time.time()
        if current_time - start_time > max_wait_time:
            logging.error(f"Task {task_id}: Polling timed out after {max_wait_time} seconds.")
            return {"status": "error", "message": f"Polling timed out after {max_wait_time} seconds"}

        try:
            response = requests.get(status_url, timeout=short_timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            status = result.get("status")

            # Check for terminal states OR processing_complete (signal to start copy)
            if status in ["completed", "failed", "copy_failed", "processing_complete"]:
                logging.info(f"Task {task_id} reached decisive status: {status}. Returning API result.")
                return result # Exit loop and return the API's result
            elif status in ["pending", "running"]:
                logging.debug(f"Task {task_id} status: {status}. Continuing to poll...")
                pass # Continue loop after sleep
            else:
                logging.error(f"Task {task_id}: Received unexpected status '{status}'. API Result: {result}")
                return {"status": "error", "message": f"Unexpected status '{status}' received from API"}

        except requests.exceptions.Timeout:
            logging.warning(f"Task {task_id}: Status check timed out (timeout={short_timeout}s). Retrying after {poll_interval}s...")
        except requests.exceptions.RequestException as e:
            logging.error(f"Task {task_id}: Error checking status: {e}")
            # Decide whether to retry or fail immediately on connection errors
            # For now, retry after a delay
            logging.warning(f"Task {task_id}: Connection error during status check. Retrying after {poll_interval}s...")

        time.sleep(poll_interval)

def _rename_output_files(final_output_dir: str, mode: str) -> Optional[str]:
    """
    在最终用户输出目录中查找、重命名文件，并返回主显示图像的预期路径。
    Args:
        final_output_dir: 用户的最终本地输出目录路径。
        mode: 处理模式 ('single_image', 'single_raster' 等)。
    Returns:
        主显示图像（掩码）的预期完整路径 (例如 '.../detection_mask.png')，
        如果找不到源掩码文件则为 None。
    """
    expected_display_path = None # 存储预期的最终显示路径
    source_mask_path = None     # 存储找到的原始掩码文件路径
    source_vis_path = None      # 存储找到的原始可视化文件路径
    source_result_tif_path = None # 存储找到的原始栅格结果文件路径

    target_mask_name = "detection_mask.png" if mode in ["single_image", "batch_image"] else "detection_mask.tif"
    target_vis_name = "detection_visualization.png"
    target_result_tif_name = "detection_result.tif"

    logging.info(f"Starting rename process in directory: {final_output_dir} for mode: {mode}")

    try:
        if not os.path.isdir(final_output_dir):
            logging.warning(f"Final output directory does not exist: {final_output_dir}")
            return None

        items = os.listdir(final_output_dir)
        logging.debug(f"Items found in {final_output_dir}: {items}")

        # --- Pass 1: Identify source files ---
        for item_name in items:
            item_path = os.path.join(final_output_dir, item_name)
            if not os.path.isfile(item_path):
                continue

            item_name_lower = item_name.lower()

            # Identify potential source mask file
            if mode in ["single_image", "batch_image"] and item_name_lower.endswith("_result.png"):
                source_mask_path = item_path
                logging.debug(f"Identified potential image mask source: {source_mask_path}")
            elif mode in ["single_raster", "batch_raster"] and item_name_lower.endswith("_mask.tif"):
                source_mask_path = item_path
                logging.debug(f"Identified potential raster mask source: {source_mask_path}")

            # Identify potential source visualization file
            elif "quadview" in item_name_lower and item_name_lower.endswith(".png"):
                 source_vis_path = item_path
                 logging.debug(f"Identified potential visualization source: {source_vis_path}")

            # Identify potential source secondary raster result file
            elif mode in ["single_raster", "batch_raster"] and item_name_lower.endswith("_result.tif"):
                 source_result_tif_path = item_path
                 logging.debug(f"Identified potential secondary raster result source: {source_result_tif_path}")

        # --- Pass 2: Determine expected display path and attempt renames ---
        rename_ops = []
        if source_mask_path:
            expected_display_path = os.path.join(final_output_dir, target_mask_name)
            rename_ops.append((source_mask_path, expected_display_path))
        else:
             logging.warning(f"Could not find a source mask file (_result.png or _mask.tif) in {final_output_dir}")

        if source_vis_path:
            target_vis_path = os.path.join(final_output_dir, target_vis_name)
            rename_ops.append((source_vis_path, target_vis_path))

        if source_result_tif_path:
            target_result_tif_path = os.path.join(final_output_dir, target_result_tif_name)
            rename_ops.append((source_result_tif_path, target_result_tif_path))

        renamed_files_log = []
        for source, target in rename_ops:
            # Skip if source and target are somehow the same
            if os.path.exists(source) and os.path.normcase(source) == os.path.normcase(target):
                continue
                
            try:
                # Check if target exists and is not the same file
                if os.path.exists(target):
                     logging.warning(f"Target file {target} already exists. Skipping rename for {source}.")
                elif os.path.exists(source):
                    os.rename(source, target)
                    renamed_files_log.append(f"{os.path.basename(source)} -> {os.path.basename(target)}")
                    logging.info(f"Successfully renamed {os.path.basename(source)} to {os.path.basename(target)}")
                else:
                     logging.warning(f"Source file {source} not found for renaming.")
            except OSError as e:
                logging.warning(f"Could not rename {os.path.basename(source)} to {os.path.basename(target)}: {e}")
            except Exception as e:
                 logging.error(f"Unexpected error renaming {os.path.basename(source)}: {e}")


        if renamed_files_log:
            logging.info(f"Rename operations completed in {final_output_dir}. Files renamed: {', '.join(renamed_files_log)}")
        else:
             logging.info(f"No files were renamed in {final_output_dir}.")

        # Return the *expected* path, even if rename failed (e.g., file existed)
        logging.info(f"Rename process finished. Returning expected display path: {expected_display_path}")
        return expected_display_path

    except Exception as e:
        logging.error(f"Error during file renaming process in {final_output_dir}: {e}", exc_info=True)
        return None # Return None if the process fails critically

# --- Main Client Function ---

def detect_changes(before_path: str, after_path: str,
                  output_path: str, mode: str = "single_image") -> Dict[str, Any]:
    """
    Orchestrates the change detection process from the client side.
    Handles file copying, API interaction, polling, and result retrieval.

    Args:
        before_path: Local path to the before image/directory.
        after_path: Local path to the after image/directory.
        output_path: Local path for the final output directory.
        mode: Processing mode ('single_image', 'single_raster', 'batch_image', 'batch_raster').

    Returns:
        Dict: Final result status and information.
    """
    session_id = _generate_session_id()
    client_temp_output_dir = None # To store the path of the temporary output dir on the client

    try:
        # --- 1. Prepare Inputs and Copy to Shared Volume ---
        logging.info(f"[{session_id}] Starting detection (mode: {mode}). Before: {before_path}, After: {after_path}, Output: {output_path}")
        docker_before_path = ""
        docker_after_path = ""
        docker_output_path = ""

        if mode in ["single_image", "single_raster"]:
            if not os.path.isfile(before_path): return {"status": "error", "message": f"Before path is not a file: {before_path}"}
            if not os.path.isfile(after_path): return {"status": "error", "message": f"After path is not a file: {after_path}"}
            _, docker_before_path = _copy_to_shared(before_path, T1_DIR_CLIENT, T1_DIR_DOCKER, session_id)
            _, docker_after_path = _copy_to_shared(after_path, T2_DIR_CLIENT, T2_DIR_DOCKER, session_id)
            client_temp_output_dir, docker_output_path = _prepare_output_directory(OUTPUT_DIR_CLIENT, OUTPUT_DIR_DOCKER, output_path, session_id)
        elif mode in ["batch_image", "batch_raster"]:
            if not os.path.isdir(before_path): return {"status": "error", "message": f"Before path is not a directory: {before_path}"}
            if not os.path.isdir(after_path): return {"status": "error", "message": f"After path is not a directory: {after_path}"}
            _, docker_before_path = _copy_directory_to_shared(before_path, T1_DIR_CLIENT, T1_DIR_DOCKER, session_id)
            _, docker_after_path = _copy_directory_to_shared(after_path, T2_DIR_CLIENT, T2_DIR_DOCKER, session_id)
            client_temp_output_dir, docker_output_path = _prepare_output_directory(OUTPUT_DIR_CLIENT, OUTPUT_DIR_DOCKER, output_path, session_id)
        else:
            return {"status": "error", "message": f"Unsupported mode: {mode}"}

        # --- 2. Call API to Start Detection Task ---
        api_endpoint = f"{API_BASE_URL}/detect/{mode}"
        payload = {
            "mode": mode, # Although redundant in URL, send in payload too
            "before_path": docker_before_path, # DOCKER path
            "after_path": docker_after_path,   # DOCKER path
            "output_path": docker_output_path  # DOCKER path for API output
        }
        logging.info(f"[{session_id}] Sending request to API: {api_endpoint} with payload: {payload}")
        response = requests.post(api_endpoint, json=payload, timeout=30)
        response.raise_for_status() # Check for HTTP errors

        api_start_result = response.json()
        task_id = api_start_result.get("task_id")
        if not task_id:
            logging.error(f"[{session_id}] API did not return a valid task_id. Response: {api_start_result}")
            return {"status": "error", "message": "API did not return a valid task_id", "detail": api_start_result}
        logging.info(f"[{session_id}] Task created successfully. Task ID: {task_id}")

        # --- 3. Wait for Task Completion ---
        task_result_api = _wait_for_task_completion(task_id)
        api_status = task_result_api.get("status")
        api_message = task_result_api.get("message", "")

        # Handle API-side failures reported via polling
        if api_status not in ["processing_complete", "completed"]: # completed might be returned if API handles copying, but our client expects processing_complete
            logging.error(f"[{session_id}] Task {task_id} failed or ended unexpectedly on API side. Status: {api_status}, Message: {api_message}")
            # Return the error reported by the API
            return task_result_api # This dict already contains status='error' or 'failed' etc.

        # --- 4. Process Results (Copy from Shared Volume and Rename) ---
        logging.info(f"[{session_id}] Task {task_id} processing complete (API Status: {api_status}). Starting result retrieval and renaming.")

        if not client_temp_output_dir or not os.path.isdir(client_temp_output_dir):
             logging.error(f"[{session_id}] Client temporary output directory not found or invalid: {client_temp_output_dir}")
             return {"status": "copy_failed", "message": f"Client temporary output directory not found: {client_temp_output_dir}", "task_id": task_id}

        final_display_image_path = None
        try:
            # Ensure the final user output directory exists
            os.makedirs(output_path, exist_ok=True)
            # Copy results from the temporary shared client dir to the final user dir
            logging.info(f"[{session_id}] Copying results from {client_temp_output_dir} to {output_path}")
            # Use copytree with dirs_exist_ok=True to merge contents if the final dir already exists
            shutil.copytree(client_temp_output_dir, output_path, dirs_exist_ok=True)

            # Rename files in the final user directory
            logging.info(f"[{session_id}] Renaming output files in {output_path}")
            final_display_image_path = _rename_output_files(output_path, mode)

            # --- 5. Construct Final Success Result ---
            final_status = "completed"
            final_message = f"Task completed successfully. Results saved to: {output_path}"
            logging.info(f"[{session_id}] {final_message}")
            return {
                "status": final_status,
                "message": final_message,
                "task_id": task_id,
                "output_path": output_path, # Final user path
                "display_image_path": final_display_image_path # Renamed path in user dir
                # Optionally include original API result details if needed
                # "api_result": task_result_api.get("result", {})
            }

        except Exception as copy_rename_e:
            error_msg = f"Failed during result copying or renaming: {str(copy_rename_e)}"
            logging.error(f"[{session_id}] {error_msg}", exc_info=True)
            return {
                "status": "copy_failed", # Specific status for client-side copy/rename failure
                "message": error_msg,
                "task_id": task_id,
                "output_path": output_path, # Still report intended output path
                "display_image_path": final_display_image_path # May be None or partially renamed path
            }
        # --- Clean up the temporary client output directory ---
        # finally:
        #     if client_temp_output_dir and os.path.isdir(client_temp_output_dir):
        #         try:
        #             logging.info(f"[{session_id}] Cleaning up temporary output directory: {client_temp_output_dir}")
        #             shutil.rmtree(client_temp_output_dir)
        #         except Exception as cleanup_e:
        #             logging.warning(f"[{session_id}] Failed to cleanup temporary directory {client_temp_output_dir}: {cleanup_e}")


    except Exception as e:
        error_message = f"An error occurred in detect_changes: {str(e)}"
        logging.error(f"[{session_id}] {error_message}", exc_info=True)
        return {"status": "error", "message": error_message}


# --- Connection and Configuration Functions ---

def check_connection() -> bool:
    """检查API连接状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.warning(f"API connection check failed: {e}")
        return False

def set_api_base_url(url: str):
    """设置API基础URL"""
    global API_BASE_URL
    API_BASE_URL = url
    logging.info(f"API Base URL set to: {url}")

def get_api_base_url() -> str:
    """获取当前API基础URL"""
    return API_BASE_URL
