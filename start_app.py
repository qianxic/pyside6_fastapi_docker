import os
import sys
import subprocess
import shutil
from pathlib import Path
import time
import PySide6
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal, QThread, Slot
import traceback

try:
    from zhuyaogongneng_docker.app import RemoteSensingApp
except ImportError:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    from zhuyaogongneng_docker.app import RemoteSensingApp

try:
    import change3d_api_docker.path_connector
except ImportError:
    pass

class Worker(QThread):
    update_status = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, task_func, task_name):
        super().__init__()
        self.task_func = task_func
        self.task_name = task_name

    def run(self):
        try:
            self.update_status.emit(f"正在执行: {self.task_name}...")
            success = self.task_func()
            self.update_status.emit(f"{self.task_name} 完成.")
            self.finished.emit(success, self.task_name)
        except Exception as e:
            error_msg = f"{self.task_name} 执行出错: {str(e)}"
            print(f"[错误] {error_msg}")
            traceback.print_exc()
            self.update_status.emit(error_msg)
            self.finished.emit(False, self.task_name)

def setup_shared_directories():
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 修改路径到正确的位置
        t1_dir = os.path.join(project_root, "change3d_api_docker", "t1")
        t2_dir = os.path.join(project_root, "change3d_api_docker", "t2")
        output_dir = os.path.join(project_root, "change3d_api_docker", "output")
        
        # 确保目录存在并设置权限
        for dir_path in [t1_dir, t2_dir, output_dir]:
            os.makedirs(dir_path, exist_ok=True)
            # 设置完全访问权限
            os.chmod(dir_path, 0o777)
            print(f"[信息] 创建并设置目录权限: {dir_path}")
        
        # 清理现有文件
        for dir_to_clear in [t1_dir, t2_dir, output_dir]:
            if os.path.isdir(dir_to_clear):
                for item in os.listdir(dir_to_clear):
                    item_path = os.path.join(dir_to_clear, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"[警告] 清理文件 {item_path} 时出错: {str(e)}")
                        
        print("[信息] 共享目录设置完成")
        return True
    except Exception as e:
        print(f"[错误] 设置共享目录时出错: {str(e)}")
        return False

def run_app():
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), "plugins")
    os.environ["QT_PLUGIN_PATH"] = plugin_path
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugin_path, "platforms")
    app = QApplication(sys.argv)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 添加change3d_api_docker到Python路径
    change3d_api_path = os.path.join(project_root, "change3d_api_docker")
    if change3d_api_path not in sys.path:
        sys.path.insert(0, change3d_api_path)
        
    zhuyaogongneng_path = os.path.join(project_root, "zhuyaogongneng_docker")
    if zhuyaogongneng_path not in sys.path:
        sys.path.insert(0, zhuyaogongneng_path)

    print("创建主窗口...")
    window = RemoteSensingApp()
    home_page = window.home_page

    print("准备后台任务...")
    cleanup_worker = Worker(setup_shared_directories, "清理缓存目录")

    cleanup_worker.update_status.connect(home_page.update_loading_message)
    cleanup_worker.finished.connect(home_page.handle_task_completion)

    print("启动后台任务...")
    cleanup_worker.start()

    print("显示主窗口...")
    window.show()

    print("启动 Qt 事件循环...")
    exit_code = app.exec()
    print("Qt 事件循环结束。")

    sys.exit(exit_code)


if __name__ == "__main__":
    run_app()