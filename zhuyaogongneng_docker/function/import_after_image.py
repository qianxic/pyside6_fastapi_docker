import os
from PySide6.QtWidgets import QFileDialog

# 不再在这里尝试导入，直接使用预加载模块
# 检查全局环境中是否存在预加载的path_connector
import sys
try:
    from change3d_api_docker.path_connector import path_connector
except ImportError:
    # 如果导入失败，检查全局环境
    if 'GLOBAL_PATH_CONNECTOR' in globals():
        path_connector = globals()['GLOBAL_PATH_CONNECTOR']
    elif 'GLOBAL_PATH_CONNECTOR' in sys.modules['__main__'].__dict__:
        path_connector = sys.modules['__main__'].__dict__['GLOBAL_PATH_CONNECTOR']
    else:
        try:
            from change3d_api_docker.path_connector import path_connector
        except ImportError:
            path_connector = None

class ImportAfterImage:
    def __init__(self, navigation_functions):
        """
        初始化导入后时相影像模块
        
        Args:
            navigation_functions: NavigationFunctions实例，用于日志记录和图像显示
        """
        self.navigation_functions = navigation_functions
        # 直接使用全局预加载的API
        self.api_available = path_connector is not None
    
    def import_after_image(self):
        """导入后时相影像"""
        # 保存原始log_message以便稍后恢复
        original_log_message = self.navigation_functions.log_message

        # 覆盖log_message方法以防止START级别的日志
        def filtered_log_message(message, level="INFO"):
            # 跳过"开始导入后时相影像"消息
            if level == "START" and "导入后时相影像" in message:
                return
            # 其他消息正常记录
            original_log_message(message, level)
            
        # 替换为过滤版本的log_message
        self.navigation_functions.log_message = filtered_log_message
        
        try:
            options = QFileDialog.Options()
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                None,
                "选择后时相影像文件",
                "",
                "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff);;所有文件 (*)",
                options=options
            )
            
            if file_path:
                # 直接使用用户选择的路径
                self.navigation_functions.file_path_after = file_path
                
                # 更新图像显示
                self.navigation_functions.update_image_display(is_before=False)
                self.navigation_functions.log_message("后时相影像导入完成", "COMPLETE")
                
                # 不再通知API路径已更新
                return self.navigation_functions.file_path_after
            else:
                self.navigation_functions.log_message("未选择文件", "INFO")
            
            return None
        finally:
            # 恢复原始log_message方法
            self.navigation_functions.log_message = original_log_message
    
    def save_image_to_dir(self, source_path, prefix=""):
        """
        保存图像到数据目录
        
        Args:
            source_path: 源文件路径
            prefix: 文件名前缀
            
        Returns:
            str: 保存后的文件路径，如果保存失败则返回None
        """
        try:
            # 获取文件名和扩展名
            file_name = os.path.basename(source_path)
            
            # 添加前缀（如果有）
            if prefix:
                file_name = f"{prefix}_{file_name}"
                
            # 构建目标路径
            # 使用源文件所在目录作为数据目录
            data_dir = os.path.dirname(source_path)
            target_path = os.path.join(data_dir, file_name)
            
            # 如果源路径和目标路径相同，则不需要复制
            if os.path.normpath(source_path) == os.path.normpath(target_path):
                return source_path
                
            # 复制文件
            import shutil
            shutil.copy2(source_path, target_path)
            
            return target_path
            
        except Exception as e:
            self.navigation_functions.log_message(f"保存图像失败: {str(e)}", "ERROR")
            return None 