import os
import glob
import logging
import shutil
from PySide6.QtWidgets import QLabel, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

# 导入ZoomableLabel类
try:
    from zhuyaogongneng_docker.display import ZoomableLabel
except ImportError:
    try:
        from display import ZoomableLabel
    except ImportError:
        # 如果无法导入，定义一个简单的类检查函数
        class ZoomableLabel:
            pass

class ClearTask:
    def __init__(self, navigation_functions, text_log):
        """
        初始化清除任务模块
        
        Args:
            navigation_functions: NavigationFunctions实例，用于日志记录和图像显示
            text_log: 日志文本区域
        """
        self.navigation_functions = navigation_functions
        self.text_log = text_log
        
    def clear_interface(self):
        """清除界面显示的所有内容"""
        try:
            # 先重置文件路径和图像信息，再清除图像显示
            # 重置导航功能类的文件路径和图像信息
            if hasattr(self.navigation_functions, 'file_path'):
                self.navigation_functions.file_path = None
            if hasattr(self.navigation_functions, 'file_path_after'):
                self.navigation_functions.file_path_after = None
            # 重置原始图像尺寸信息
            if hasattr(self.navigation_functions, 'before_image_original_size'):
                self.navigation_functions.before_image_original_size = None
            if hasattr(self.navigation_functions, 'after_image_original_size'):
                self.navigation_functions.after_image_original_size = None
            # 重置解析结果
            if hasattr(self.navigation_functions, 'result_image'):
                self.navigation_functions.result_image = None
            if hasattr(self.navigation_functions, 'result_image_path'):
                self.navigation_functions.result_image_path = None
            # 重置缓存结果
            if hasattr(self.navigation_functions, 'cached_result_image'):
                self.navigation_functions.cached_result_image = None
            # 重置掩膜图像路径
            if hasattr(self.navigation_functions, 'mask_image_path'):
                self.navigation_functions.mask_image_path = None
            if hasattr(self.navigation_functions, 'boundary_image_path'):
                self.navigation_functions.boundary_image_path = None
            # 重置标准化状态
            if hasattr(self.navigation_functions, 'is_image_standardized'):
                self.navigation_functions.is_image_standardized = False
            # 重置渔网掩膜状态 
            if hasattr(self.navigation_functions, 'has_fishnet_mask'):
                self.navigation_functions.has_fishnet_mask = False
            
            # 清除图像显示
            try:
                self._reset_labels()
            except Exception as e:
                logging.error(f"重置标签时出错: {str(e)}")
                self.navigation_functions.log_message(f"重置标签时出错: {str(e)}", "ERROR")
            
            # 清除日志区域
            if hasattr(self.text_log, 'clear') and self.text_log is not None:
                try:
                    self.text_log.clear()
                except Exception as e:
                    logging.error(f"清除日志区域出错: {str(e)}")
            
            # 确保UI更新
            QApplication.processEvents()
            
            # 清除API图片连接 - 通知API路径已更新为空
            try:
                # 检查API路径连接器是否可用
                from change3d_api_docker.path_connector import path_connector
                if path_connector is not None:
                    # 清除API相关信息
                    if hasattr(self.navigation_functions, 'notify_api_path_updated'):
                        # 将所有路径设为None，通知API清除路径连接
                        self.navigation_functions.notify_api_path_updated('before_image_path', None)
                        self.navigation_functions.notify_api_path_updated('after_image_path', None)
                        self.navigation_functions.notify_api_path_updated('result_image_path', None)
                        self.navigation_functions.log_message("已清除API图像连接", "INFO")
            except Exception as e:
                logging.warning(f"清除API图像连接时出错: {str(e)}")
            
            # 删除临时保存的结果文件
            try:
                temp_files = ['result_image_path', 'mask_image_path', 'boundary_image_path']
                for attr in temp_files:
                    file_path = getattr(self.navigation_functions, attr, None)
                    if file_path and isinstance(file_path, str) and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logging.info(f"已删除临时文件: {file_path}")
                        except Exception as e:
                            logging.warning(f"删除临时文件失败: {file_path}, 错误: {str(e)}")
            except Exception as e:
                logging.error(f"删除临时文件过程中出错: {str(e)}")
            
            # 清理目录中所有临时缓存文件
            try:
                self.clean_temp_files()
            except Exception as e:
                logging.error(f"清理临时文件过程中出错: {str(e)}")
                self.navigation_functions.log_message(f"清理临时文件时出错: {str(e)}", "ERROR")
            
            # 清除历史记录
            if hasattr(self.navigation_functions, 'clear_history'):
                try:
                    self.navigation_functions.clear_history()
                except Exception as e:
                    logging.error(f"清除历史记录时出错: {str(e)}")
            
            # 确保UI更新
            QApplication.processEvents()
            
            # 添加清除完成的消息到日志
            self.navigation_functions.log_message("界面已清空", "COMPLETE")
            
        except Exception as e:
            logging.error(f"清除界面时出错: {str(e)}")
            self.navigation_functions.log_message(f"清除界面时出错: {str(e)}", "ERROR")
            import traceback
            logging.error(traceback.format_exc())
            self.navigation_functions.log_message("清除界面失败，请尝试重启应用程序", "ERROR")
    
    def clean_temp_files(self):
        """清理所有临时缓存文件，但不清除用户选择的输出目录中的内容"""
        try:
            # 确保标签对象完全重置为初始状态
            self._reset_labels()
            
            # 用户选择的输出目录 - 这些目录中的文件不会被删除
            user_selected_dirs = []
            
            # 检查是否有用户选择的网格裁剪输出目录
            if hasattr(self.navigation_functions, 'grid_output_dir') and self.navigation_functions.grid_output_dir:
                user_selected_dirs.append(self.navigation_functions.grid_output_dir)
            
            # 检查是否有用户选择的批处理输出目录
            if hasattr(self.navigation_functions, 'batch_output_dir') and self.navigation_functions.batch_output_dir:
                user_selected_dirs.append(self.navigation_functions.batch_output_dir)
                
            # 添加其他可能的用户选择目录（如从对话框中选择的目录）
            if hasattr(self.navigation_functions, 'user_selected_output_dirs') and self.navigation_functions.user_selected_output_dirs:
                for dir_path in self.navigation_functions.user_selected_output_dirs:
                    if dir_path:
                        user_selected_dirs.append(dir_path)
            
            # 将相对路径转换为绝对路径
            user_selected_dirs = [os.path.abspath(p) if p else None for p in user_selected_dirs]
            user_selected_dirs = [p for p in user_selected_dirs if p]  # 移除None值
            
            # 清理变化检测结果临时文件
            temp_patterns = [
                "change_detection_result*.{jpg,png,tif,tiff}",
                "change_detection_mask*.{jpg,png,tif,tiff}",
                "change_detection_boundary*.{jpg,png,tif,tiff}",
                "grid_preview*.{jpg,png,tif,tiff}",
                "before_grid*.{jpg,png,tif,tiff}",
                "after_grid*.{jpg,png,tif,tiff}",
                "standardized_*.{jpg,png,tif,tiff}",
                "temp_*.{jpg,png,tif,tiff}",
                "*_preview.{jpg,png,tif,tiff}",  # 匹配所有预览图像
                "*_result*.{jpg,png,tif,tiff}", # 匹配变化检测结果图像
            ]
            
            files_removed = 0
            
            # 创建临时文件清理目录列表
            temp_dirs = [
                ".", # 当前目录
                "./temp", # 临时目录
                os.path.join(os.getcwd(), "temp"), # 绝对路径临时目录
            ]
            
            # 如果存在其他工作目录，也添加进来
            if hasattr(self.navigation_functions, 'file_path') and self.navigation_functions.file_path:
                temp_dirs.append(os.path.dirname(self.navigation_functions.file_path))
            if hasattr(self.navigation_functions, 'file_path_after') and self.navigation_functions.file_path_after:
                temp_dirs.append(os.path.dirname(self.navigation_functions.file_path_after))
            if hasattr(self.navigation_functions, 'result_image_path') and self.navigation_functions.result_image_path:
                temp_dirs.append(os.path.dirname(self.navigation_functions.result_image_path))
            
            # 将所有有效的目录添加到清理列表 - 但不包括用户选择的输出目录
            temp_dirs = [os.path.abspath(p) if p else None for p in temp_dirs]
            temp_dirs = [p for p in temp_dirs if p and p not in user_selected_dirs]  # 排除用户选择的目录
            
            # 移除重复的目录
            temp_dirs = list(set(temp_dirs))
            
            # 在每个目录中搜索并清理临时文件
            for temp_dir in temp_dirs:
                if not os.path.exists(temp_dir):
                    continue
                    
                for pattern in temp_patterns:
                    # 在当前目录查找匹配的文件
                    for file_path in glob.glob(os.path.join(temp_dir, pattern), recursive=False):
                        try:
                            # 检查文件路径是否在用户选择的目录中
                            file_abs_path = os.path.abspath(file_path)
                            file_dir = os.path.dirname(file_abs_path)
                            
                            if any(file_dir.startswith(user_dir) for user_dir in user_selected_dirs if user_dir):
                                # 跳过用户选择目录中的文件
                                logging.info(f"跳过用户选择目录中的文件: {file_path}")
                                continue
                                
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                logging.info(f"已清理临时文件: {file_path}")
                                files_removed += 1
                        except Exception as e:
                            logging.warning(f"清理临时文件失败: {file_path}, 错误: {str(e)}")
            
            # 检查是否存在临时目录，并清空它（但不处理用户选择的目录）
            temp_directory = os.path.join(os.getcwd(), "temp")
            temp_directory_abs = os.path.abspath(temp_directory)
            if os.path.exists(temp_directory) and os.path.isdir(temp_directory) and \
               not any(temp_directory_abs.startswith(user_dir) for user_dir in user_selected_dirs if user_dir):
                try:
                    shutil.rmtree(temp_directory)
                    os.makedirs(temp_directory)  # 重新创建一个空目录
                    logging.info(f"已清空并重建临时目录: {temp_directory}")
                    files_removed += 1
                except Exception as e:
                    logging.warning(f"清空临时目录失败: {temp_directory}, 错误: {str(e)}")
            
            # 清理__pycache__目录中的缓存文件
            pycache_dirs = ["__pycache__", "*/__pycache__"]
            for cache_dir_pattern in pycache_dirs:
                for cache_dir in glob.glob(cache_dir_pattern):
                    if os.path.isdir(cache_dir):
                        # 只清理目录中的文件，不删除目录本身
                        for cache_file in glob.glob(f"{cache_dir}/*.pyc") + glob.glob(f"{cache_dir}/*.pyo"):
                            try:
                                os.remove(cache_file)
                                logging.info(f"已清理Python缓存文件: {cache_file}")
                                files_removed += 1
                            except Exception as e:
                                logging.warning(f"清理缓存文件失败: {cache_file}, 错误: {str(e)}")
            
            if files_removed > 0:
                logging.info(f"共清理 {files_removed} 个临时文件")
                self.navigation_functions.log_message(f"已清理 {files_removed} 个临时文件，保留了用户选择目录中的内容")
            else:
                logging.info("没有临时文件需要清理")
                
        except Exception as e:
            logging.error(f"清理临时文件过程中发生错误: {str(e)}")
            self.navigation_functions.log_message(f"清理临时文件时出错: {str(e)}")

    def _reset_labels(self):
        """重置标签对象为初始状态"""
        if not hasattr(self.navigation_functions, 'label_before') or \
           not hasattr(self.navigation_functions, 'label_after') or \
           not hasattr(self.navigation_functions, 'label_result'):
           self.navigation_functions.log_message("无法重置标签：缺少必要的标签引用", "ERROR")
           return
        
        # 清除前时相标签内容
        if hasattr(self.navigation_functions.label_before, 'clear'):
            self.navigation_functions.label_before.clear()
            # 确保UI更新
            if hasattr(self.navigation_functions.label_before, 'update'):
                self.navigation_functions.label_before.update()
            # 如果是普通QLabel，设置提示文本
            if isinstance(self.navigation_functions.label_before, QLabel) and not isinstance(self.navigation_functions.label_before, ZoomableLabel):
                self.navigation_functions.label_before.setText("前时相影像")
                
        # 清除后时相标签内容
        if hasattr(self.navigation_functions.label_after, 'clear'):
            self.navigation_functions.label_after.clear()
            # 确保UI更新
            if hasattr(self.navigation_functions.label_after, 'update'):
                self.navigation_functions.label_after.update()
            # 如果是普通QLabel，设置提示文本
            if isinstance(self.navigation_functions.label_after, QLabel) and not isinstance(self.navigation_functions.label_after, ZoomableLabel):
                self.navigation_functions.label_after.setText("后时相影像")
        
        # 清除输出标签内容
        if hasattr(self.navigation_functions.label_result, 'clear'):
            self.navigation_functions.label_result.clear()
            # 确保UI更新
            if hasattr(self.navigation_functions.label_result, 'update'):
                self.navigation_functions.label_result.update()
            # 如果是普通QLabel，设置提示文本
            if isinstance(self.navigation_functions.label_result, QLabel) and not isinstance(self.navigation_functions.label_result, ZoomableLabel):
                self.navigation_functions.label_result.setText("解译结果")
        
        # 查找并清除任何可能的变化检测任务实例中的结果图像
        try:
            # 尝试寻找相关的变化检测任务实例并清除其结果
            from zhuyaogongneng_docker.function.change_cd import ExecuteChangeDetectionTask
            for attr_name in dir(self.navigation_functions):
                attr = getattr(self.navigation_functions, attr_name)
                if isinstance(attr, ExecuteChangeDetectionTask):
                    if hasattr(attr, 'result_image'):
                        attr.result_image = None
                    if hasattr(attr, 'result_image_path'):
                        attr.result_image_path = None
        except ImportError:
            # 如果无法导入模块，忽略这一步
            pass
            
        # 重置NavigationFunctions中的结果图像
        if hasattr(self.navigation_functions, 'result_image'):
            self.navigation_functions.result_image = None
        if hasattr(self.navigation_functions, 'result_image_path'):
            self.navigation_functions.result_image_path = None
            
        # 强制更新显示，确保界面立即更新
        QApplication.processEvents()
