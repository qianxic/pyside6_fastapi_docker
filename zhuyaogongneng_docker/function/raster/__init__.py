"""
栅格数据处理包
提供栅格影像变化检测所需的各种功能模块
包括栅格数据读写、变化检测、批处理等功能
"""

# 版本信息
__version__ = '1.0.0'

# 导入主要类
from .detection import RasterChangeDetection
from .import_module import RasterImporter
from .batch_processor import RasterBatchProcessor
from .grid import RasterGridCropping

# 导出接口
__all__ = [
    'RasterChangeDetection',
    'RasterImporter',
    'RasterBatchProcessor',
    'RasterGridCropping'
]

# 提供向后兼容性，允许从原位置导入
import sys
import importlib

def __getattr__(name):
    """允许懒加载模块"""
    if name in __all__:
        return globals()[name]
    elif name == 'ShapefileGenerator' or name == 'RasterExporter':
        # 提供兼容层，返回None对象或替代实现
        return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") 