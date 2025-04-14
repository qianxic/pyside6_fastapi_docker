"""
主题工具模块 - 提供对主题的访问功能
"""
import os
import sys
import importlib.util

def get_theme_manager():
    """
    加载并返回ThemeManager类
    
    这个函数解决了导入ThemeManager的路径问题，
    不管调用者在哪个路径，都能正确导入ThemeManager
    
    Returns:
        ThemeManager类
    """
    # 获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file)
    
    # 获取父目录
    parent_dir = os.path.dirname(current_dir)
    
    # theme_manager.py的路径
    theme_manager_path = os.path.join(parent_dir, 'theme_manager.py')
    
    # 检查文件是否存在
    if not os.path.exists(theme_manager_path):
        raise FileNotFoundError(f"找不到theme_manager.py文件: {theme_manager_path}")
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("theme_manager", theme_manager_path)
    theme_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theme_module)
    
    # 返回ThemeManager类
    return theme_module.ThemeManager

# 为了方便直接使用，预先加载ThemeManager类
ThemeManager = get_theme_manager() 