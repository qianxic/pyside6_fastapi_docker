import os
from PySide6.QtWidgets import QFileDialog

# 尝试导入路径连接器
try:
    from change3d_api_docker.path_connector import path_connector
except ImportError:
    try:
        from change3d_api_docker.path_connector import path_connector
    except ImportError:
        path_connector = None

class ImportBeforeImage:
    def __init__(self, navigation_functions):
        """
        初始化导入前时相影像模块
        
        Args:
            navigation_functions: NavigationFunctions实例，用于日志记录和图像显示
        """
        self.navigation_functions = navigation_functions
        # 检查API可用性
        self.api_available = path_connector is not None
    
    def on_import_clicked(self):
        """导入前时相影像"""
        
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            None,
            "选择前时相影像文件",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff);;所有文件 (*)",
            options=options
        )
        
        if file_path:
            self.navigation_functions.log_message(f"已选择前时相影像: {file_path}", "INFO")
            
            # 直接使用用户选择的路径
            self.navigation_functions.file_path = file_path
            
            # 更新图像显示
            self.navigation_functions.update_image_display()
            self.navigation_functions.log_message("前时相影像导入完成", "COMPLETE")
            
            # 不再通知API路径已更新
            return self.navigation_functions.file_path
        else:
            self.navigation_functions.log_message("未选择文件", "INFO")
            
        return None
    
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