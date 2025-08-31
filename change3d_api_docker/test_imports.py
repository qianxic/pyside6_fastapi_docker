#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模块导入的脚本
"""

import sys
from pathlib import Path

def test_imports():
    """测试所有必要的模块导入"""
    print("=== 开始测试模块导入 ===")
    
    # 设置路径
    current_dir = Path(__file__).parent.absolute()
    project_dir = current_dir.parent.absolute()
    change3d_docker_path = project_dir / "change3d_docker"
    
    print(f"当前目录: {current_dir}")
    print(f"项目目录: {project_dir}")
    print(f"change3d_docker路径: {change3d_docker_path}")
    print(f"change3d_docker存在: {change3d_docker_path.exists()}")
    
    # 添加路径
    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(current_dir))
    if change3d_docker_path.exists():
        sys.path.insert(0, str(change3d_docker_path))
    
    print(f"Python路径: {sys.path}")
    
    # 测试导入
    try:
        print("\n1. 测试基础模块导入...")
        import torch
        print(f"✓ torch导入成功，版本: {torch.__version__}")
        
        print("\n2. 测试change3d_docker模块导入...")
        import change3d_docker
        print("✓ change3d_docker导入成功")
        
        print("\n3. 测试scripts_app模块导入...")
        from change3d_docker.scripts_app import large_image_BCD
        print("✓ large_image_BCD导入成功")
        
        from change3d_docker.scripts_app import large_raster_BCD
        print("✓ large_raster_BCD导入成功")
        
        from change3d_docker.scripts_app import batch_image_BCD
        print("✓ batch_image_BCD导入成功")
        
        from change3d_docker.scripts_app import batch_raster_BCD
        print("✓ batch_raster_BCD导入成功")
        
        print("\n4. 测试change_detection_model导入...")
        from change_detection_model import detection_model, DetectionMode
        print("✓ change_detection_model导入成功")
        
        print("\n5. 测试FastAPI相关导入...")
        from fastapi import FastAPI
        print("✓ FastAPI导入成功")
        
        print("\n=== 所有模块导入测试通过 ===")
        return True
        
    except Exception as e:
        print(f"\n❌ 导入失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
