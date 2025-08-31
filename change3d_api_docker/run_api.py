import os
import sys
import subprocess
import argparse
from pathlib import Path

#http://127.0.0.1:8000/docs#/

def setup_directories():
    """设置必要的目录结构"""
    # 确保t1、t2、output目录存在
    current_dir = Path(__file__).parent.absolute()
    
    t1_dir = current_dir / "t1"
    t2_dir = current_dir / "t2"
    output_dir = current_dir / "output"
    
    # 创建目录
    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"目录结构已准备就绪:")
    print(f"  前时相目录: {t1_dir}")
    print(f"  后时相目录: {t2_dir}")
    print(f"  输出目录: {output_dir}")

def main():
    """启动API服务"""
    # 设置环境变量解决OpenMP冲突问题
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # 设置目录结构
    setup_directories()
    
    # 获取当前目录
    current_dir = Path(__file__).parent.absolute()
    
    project_dir = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_dir))
    sys.path.insert(0, str(current_dir))
    
    # 确保change3d_docker模块可以被找到
    change3d_docker_path = project_dir / "change3d_docker"
    if change3d_docker_path.exists():
        sys.path.insert(0, str(change3d_docker_path))
    
    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    print(f"当前目录中的文件: {os.listdir(current_dir)}")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动遥感影像变化检测API服务")
    parser.add_argument("--host", default="127.0.0.1", help="绑定主机地址")
    parser.add_argument("--port", default=8000, type=int, help="绑定端口")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    parser.add_argument("--workers", default=1, type=int, help="工作进程数")
    args = parser.parse_args()
    
    # 构建命令
    cmd = [
        "uvicorn", 
        "main:app", 
        f"--host={args.host}", 
        f"--port={args.port}"
    ]
    
    # 添加可选参数
    if args.reload:
        cmd.append("--reload")
    if args.workers > 1:
        cmd.append(f"--workers={args.workers}")
    
    # 打印启动信息
    print(f"正在启动API服务: {' '.join(cmd)}")
    print(f"API文档将可通过 http://{args.host}:{args.port}/docs 访问")
    
    # 启动服务
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("API服务已停止")
    except Exception as e:
        print(f"启动API服务时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 