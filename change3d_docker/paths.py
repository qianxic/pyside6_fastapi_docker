#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 项目路径管理模块 - 用于统一管理项目中的所有路径

import os
import sys
from pathlib import Path

def find_project_root(path=None):
    """
    从给定路径向上查找项目根目录
    通过查找.project_root标记文件来确定项目根目录
    """
    if path is None:
        # 使用当前文件所在目录作为起点
        path = Path(os.path.dirname(os.path.abspath(__file__)))
    else:
        path = Path(path)
    
    # 如果当前目录有标记文件，则当前目录是项目根目录
    if (path / ".project_root").exists():
        return path
    
    # 向上查找，直到找到标记文件或到达文件系统根目录
    parent = path.parent
    if parent == path:  # 到达文件系统根目录
        raise FileNotFoundError("无法找到项目根目录，请确保.project_root文件存在")
    
    # 递归向上查找
    return find_project_root(parent)

# 项目根目录
PROJECT_ROOT = find_project_root()

# 重要目录路径
SCRIPTS_APP_DIR = PROJECT_ROOT / "scripts_app"
API_DIR = PROJECT_ROOT / "change_detection_api"
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"

# 模型路径
DEFAULT_MODEL_PATH = PROJECT_ROOT / "exp_resume/LEVIR-CD_iter_15000_lr_0.0002/checkpoint.pth.tar"

# 临时目录
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

def add_project_paths_to_sys_path():
    """将项目相关路径添加到sys.path"""
    # 添加项目根目录
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # 添加scripts_app目录
    sys.path.insert(0, str(SCRIPTS_APP_DIR))
    
    # 添加项目父目录，以便可以通过change3d包名导入
    sys.path.insert(0, str(PROJECT_ROOT.parent))
    
    print(f"已添加项目路径到sys.path:\n{PROJECT_ROOT}\n{SCRIPTS_APP_DIR}\n{PROJECT_ROOT.parent}")

def setup_module_paths():
    """设置模块路径并返回一个上下文对象，包含所有重要路径"""
    add_project_paths_to_sys_path()
    
    return {
        "project_root": PROJECT_ROOT,
        "scripts_app_dir": SCRIPTS_APP_DIR,
        "api_dir": API_DIR,
        "model_dir": MODEL_DIR,
        "data_dir": DATA_DIR,
        "default_model_path": DEFAULT_MODEL_PATH,
        "temp_dir": TEMP_DIR
    }

# 如果直接运行此模块，打印所有路径信息
if __name__ == "__main__":
    paths = setup_module_paths()
    
    print("项目路径信息:")
    for name, path in paths.items():
        print(f"{name}: {path}")
        print(f"路径存在: {Path(path).exists()}")
    
    print("\n系统路径:")
    for i, path in enumerate(sys.path[:5]):
        print(f"{i}: {path}") 