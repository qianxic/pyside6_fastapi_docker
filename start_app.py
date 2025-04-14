import os
import sys
import subprocess
import shutil
from pathlib import Path
import time
import PySide6
# Import necessary Qt components for threading and signals
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal, QThread, Slot
import traceback # For detailed error logging

# Import the main app class
try:
    from zhuyaogongneng_docker.app import RemoteSensingApp # Import the main window class
except ImportError:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    from zhuyaogongneng_docker.app import RemoteSensingApp

# Import path_connector if needed elsewhere, though not directly used here
try:
    import change3d_api_docker.path_connector
except ImportError:
    pass

# --- Background Worker ---
class Worker(QThread):
    """Worker thread to run background tasks without blocking the UI."""
    update_status = Signal(str) # Signal to update status message
    finished = Signal(bool, str) # Signal emitted when task finishes (success, task_name)

    def __init__(self, task_func, task_name):
        super().__init__()
        self.task_func = task_func
        self.task_name = task_name

    def run(self):
        """Execute the task function."""
        try:
            self.update_status.emit(f"正在执行: {self.task_name}...")
            success = self.task_func() # Execute the actual function
            self.update_status.emit(f"{self.task_name} 完成.")
            self.finished.emit(success, self.task_name)
        except Exception as e:
            error_msg = f"{self.task_name} 执行出错: {str(e)}"
            print(f"[错误] {error_msg}")
            traceback.print_exc()
            self.update_status.emit(error_msg)
            self.finished.emit(False, self.task_name)

# --- Task Functions (modified slightly to return success) ---

def setup_shared_directories():
    """Ensures shared directories exist and clears them. Returns True on success."""
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        t1_dir = os.path.join(project_root, "change3d_api_docker", "t1")
        t2_dir = os.path.join(project_root, "change3d_api_docker", "t2")
        output_dir = os.path.join(project_root, "change3d_api_docker", "output")
        os.makedirs(t1_dir, exist_ok=True)
        os.makedirs(t2_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        for dir_to_clear in [t1_dir, t2_dir, output_dir]:
            if os.path.isdir(dir_to_clear):
                for item in os.listdir(dir_to_clear):
                    item_path = os.path.join(dir_to_clear, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception: pass # Ignore deletion errors
        return True # Assume success unless exception below
    except Exception as e:
        print(f"[错误] 清理共享目录时出错: {str(e)}")
        return False # Indicate failure

# --- Main Application Execution ---
def run_app():
    # Set up Qt application first
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), "plugins")
    os.environ["QT_PLUGIN_PATH"] = plugin_path
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugin_path, "platforms")
    app = QApplication(sys.argv)

    # --- Environment Setup ---
    # Change to script dir might be important for relative paths in app code
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    zhuyaogongneng_path = os.path.join(project_root, "zhuyaogongneng_docker")
    if zhuyaogongneng_path not in sys.path:
        sys.path.insert(0, zhuyaogongneng_path)

    # ---

    # Create the main window - this shows HomePage immediately
    print("创建主窗口...")
    window = RemoteSensingApp()
    home_page = window.home_page # Get reference to HomePage

    # --- Setup Background Tasks ---
    print("准备后台任务...")
    # Create workers
    cleanup_worker = Worker(setup_shared_directories, "清理缓存目录")

    # Connect signals to HomePage slots
    cleanup_worker.update_status.connect(home_page.update_loading_message)
    cleanup_worker.finished.connect(home_page.handle_task_completion)

    # Start background tasks
    print("启动后台任务...")
    cleanup_worker.start()
    # ---

    # Show the main window
    print("显示主窗口...")
    window.show()

    # Start the Qt event loop
    print("启动 Qt 事件循环...")
    exit_code = app.exec()
    print("Qt 事件循环结束。")

    # Exit with the application's exit code
    sys.exit(exit_code)


if __name__ == "__main__":
    run_app() 