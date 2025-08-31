#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的模块导入测试
"""

import os
import sys
from pathlib import Path

def simple_test():
    """简单的导入测试"""
    print("=== 简单导入测试 ===")
    
    try:
        print("1. 测试基础Python模块...")
        import os
        import sys
        print("✓ 基础模块导入成功")
        
        print("\n2. 测试pathlib...")
        from pathlib import Path
        print("✓ pathlib导入成功")
        
        print("\n3. 测试FastAPI...")
        from fastapi import FastAPI
        print("✓ FastAPI导入成功")
        
        print("\n4. 测试pydantic...")
        from pydantic import BaseModel
        print("✓ pydantic导入成功")
        
        print("\n5. 测试uvicorn...")
        import uvicorn
        print("✓ uvicorn导入成功")
        
        print("\n6. 测试torch...")
        import torch
        print(f"✓ torch导入成功，版本: {torch.__version__}")
        
        print("\n7. 测试change3d_docker...")
        # 设置路径
        current_dir = Path(__file__).parent.absolute()
        project_dir = current_dir.parent.absolute()
        change3d_docker_path = project_dir / "change3d_docker"
        
        sys.path.insert(0, str(project_dir))
        sys.path.insert(0, str(current_dir))
        if change3d_docker_path.exists():
            sys.path.insert(0, str(change3d_docker_path))
        
        import change3d_docker
        print("✓ change3d_docker导入成功")
        
        print("\n8. 测试scripts_app...")
        from change3d_docker.scripts_app import large_image_BCD
        print("✓ large_image_BCD导入成功")
        
        print("\n=== 所有测试通过 ===")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()
