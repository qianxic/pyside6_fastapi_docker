"""
功能模块包初始化文件
使用相对导入和绝对导入兼容不同的运行方式
"""
import os
import sys

# 处理不同的导入情况
try:
    # 相对导入（当作为包导入时）
    from .import_before_image import ImportBeforeImage
    from .import_after_image import ImportAfterImage
    from .change_cd import ExecuteChangeDetectionTask
    from .clear_task import ClearTask
    from .fishnet_fenge import GridCropping  # 直接从fishnet_fenge.py导入
    from .image_display import ImageDisplay
    from .batch_processing import BatchProcessing
    
except ImportError:
    # 绝对导入（当直接运行时）
    try:
        from zhuyaogongneng_docker.function.import_before_image import ImportBeforeImage
        from zhuyaogongneng_docker.function.import_after_image import ImportAfterImage
        from zhuyaogongneng_docker.function.change_cd import ExecuteChangeDetectionTask
        from zhuyaogongneng_docker.function.clear_task import ClearTask
        from zhuyaogongneng_docker.function.fishnet_fenge import GridCropping  # 直接从fishnet_fenge.py导入
        from zhuyaogongneng_docker.function.image_display import ImageDisplay
        from zhuyaogongneng_docker.function.batch_processing import BatchProcessing
        
    except ImportError as e:
        # 记录错误但不抛出异常，允许程序继续运行
        import traceback
        print(f"警告: 功能模块导入出错: {e}")
        print(traceback.format_exc())

__all__ = [
    'GridCropping',
    'ImportBeforeImage',
    'ImportAfterImage',
    'ExecuteChangeDetectionTask',
    'ClearTask',
    'ImageDisplay',
    'BatchProcessing'
]

"""
遥感影像变化检测系统包
"""
# 可以为空，只是为了标记这是一个Python包 