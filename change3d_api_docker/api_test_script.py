#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遥感影像变化检测API测试脚本
测试所有变化检测接口的功能实现
"""

import os
import sys
import time
import json
import requests
import shutil
from pathlib import Path
from typing import Dict, Any, List
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChangeDetectionAPITester:
    """变化检测API测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化测试器
        
        Args:
            base_url: API服务的基础URL
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # 测试数据路径
        self.test_data_root = Path("G:/1代码/开发/RSIIS/遥感影像变化检测系统V1.1/change3d_docker/dataes_test/val")
        self.t1_dir = self.test_data_root / "t1"
        self.t2_dir = self.test_data_root / "t2"
        self.label_dir = self.test_data_root / "label"
        
        # 测试结果输出目录
        self.test_output_dir = Path("test_outputs")
        self.test_output_dir.mkdir(exist_ok=True)
        
        # 任务状态跟踪
        self.task_results = {}
        
    def test_health_check(self) -> bool:
        """测试健康检查接口"""
        logger.info("=" * 50)
        logger.info("测试健康检查接口")
        logger.info("=" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                result = response.json()
                logger.info(f"健康检查成功: {result}")
                return True
            else:
                logger.error(f"健康检查失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"健康检查异常: {str(e)}")
            return False
    
    def test_single_image_detection(self) -> bool:
        """测试单图像变化检测"""
        logger.info("=" * 50)
        logger.info("测试单图像变化检测")
        logger.info("=" * 50)
        
        # 选择第一对图像进行测试
        t1_files = list(self.t1_dir.glob("*.png"))
        t2_files = list(self.t2_dir.glob("*.png"))
        
        if not t1_files or not t2_files:
            logger.error("测试数据不存在")
            return False
        
        # 使用第一对图像
        t1_file = t1_files[0]
        t2_file = t2_files[0]
        
        # 准备输出路径
        output_file = self.test_output_dir / f"single_image_result_{t1_file.stem}.png"
        
        # 准备请求数据
        request_data = {
            "mode": "single_image",
            "before_path": str(t1_file),
            "after_path": str(t2_file),
            "output_path": str(output_file)
        }
        
        logger.info(f"请求数据: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/detect/single_image",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"单图像检测任务创建成功: {result}")
                
                # 保存任务ID用于后续查询
                task_id = result.get("task_id")
                if task_id:
                    self.task_results["single_image"] = {
                        "task_id": task_id,
                        "request_data": request_data,
                        "response": result
                    }
                
                return True
            else:
                logger.error(f"单图像检测失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"单图像检测异常: {str(e)}")
            return False
    
    def test_single_raster_detection(self) -> bool:
        """测试单栅格影像变化检测"""
        logger.info("=" * 50)
        logger.info("测试单栅格影像变化检测")
        logger.info("=" * 50)
        
        # 选择第一对图像进行测试（作为栅格处理）
        t1_files = list(self.t1_dir.glob("*.png"))
        t2_files = list(self.t2_dir.glob("*.png"))
        
        if not t1_files or not t2_files:
            logger.error("测试数据不存在")
            return False
        
        # 使用第一对图像
        t1_file = t1_files[0]
        t2_file = t2_files[0]
        
        # 准备输出路径
        output_file = self.test_output_dir / f"single_raster_result_{t1_file.stem}.tif"
        
        # 准备请求数据
        request_data = {
            "mode": "single_raster",
            "before_path": str(t1_file),
            "after_path": str(t2_file),
            "output_path": str(output_file)
        }
        
        logger.info(f"请求数据: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/detect/single_raster",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"单栅格检测任务创建成功: {result}")
                
                # 保存任务ID用于后续查询
                task_id = result.get("task_id")
                if task_id:
                    self.task_results["single_raster"] = {
                        "task_id": task_id,
                        "request_data": request_data,
                        "response": result
                    }
                
                return True
            else:
                logger.error(f"单栅格检测失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"单栅格检测异常: {str(e)}")
            return False
    
    def test_batch_image_detection(self) -> bool:
        """测试批量图像变化检测"""
        logger.info("=" * 50)
        logger.info("测试批量图像变化检测")
        logger.info("=" * 50)
        
        # 准备输出目录
        batch_output_dir = self.test_output_dir / "batch_image_results"
        batch_output_dir.mkdir(exist_ok=True)
        
        # 准备请求数据
        request_data = {
            "mode": "batch_image",
            "before_path": str(self.t1_dir),
            "after_path": str(self.t2_dir),
            "output_path": str(batch_output_dir)
        }
        
        logger.info(f"请求数据: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/detect/batch_image",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"批量图像检测任务创建成功: {result}")
                
                # 保存任务ID用于后续查询
                task_id = result.get("task_id")
                if task_id:
                    self.task_results["batch_image"] = {
                        "task_id": task_id,
                        "request_data": request_data,
                        "response": result
                    }
                
                return True
            else:
                logger.error(f"批量图像检测失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"批量图像检测异常: {str(e)}")
            return False
    
    def test_batch_raster_detection(self) -> bool:
        """测试批量栅格影像变化检测"""
        logger.info("=" * 50)
        logger.info("测试批量栅格影像变化检测")
        logger.info("=" * 50)
        
        # 准备输出目录
        batch_output_dir = self.test_output_dir / "batch_raster_results"
        batch_output_dir.mkdir(exist_ok=True)
        
        # 准备请求数据
        request_data = {
            "mode": "batch_raster",
            "before_path": str(self.t1_dir),
            "after_path": str(self.t2_dir),
            "output_path": str(batch_output_dir)
        }
        
        logger.info(f"请求数据: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/detect/batch_raster",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"批量栅格检测任务创建成功: {result}")
                
                # 保存任务ID用于后续查询
                task_id = result.get("task_id")
                if task_id:
                    self.task_results["batch_raster"] = {
                        "task_id": task_id,
                        "request_data": request_data,
                        "response": result
                    }
                
                return True
            else:
                logger.error(f"批量栅格检测失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"批量栅格检测异常: {str(e)}")
            return False
    
    def wait_for_task_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            任务结果
        """
        logger.info(f"等待任务 {task_id} 完成...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/tasks/{task_id}")
                if response.status_code == 200:
                    task_status = response.json()
                    status = task_status.get("status")
                    
                    logger.info(f"任务状态: {status}")
                    
                    if status in ["processing_complete", "completed", "success"]:
                        logger.info(f"任务 {task_id} 完成")
                        return task_status
                    elif status in ["failed", "error"]:
                        logger.error(f"任务 {task_id} 失败: {task_status.get('message', '未知错误')}")
                        return task_status
                    elif status in ["pending", "running"]:
                        logger.info(f"任务 {task_id} 仍在执行中...")
                        time.sleep(5)  # 等待5秒后再次查询
                    else:
                        logger.warning(f"未知任务状态: {status}")
                        time.sleep(5)
                else:
                    logger.error(f"查询任务状态失败: {response.status_code} - {response.text}")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"查询任务状态异常: {str(e)}")
                time.sleep(5)
        
        logger.error(f"任务 {task_id} 超时")
        return {"status": "timeout", "message": "任务执行超时"}
    
    def test_task_status_queries(self) -> bool:
        """测试任务状态查询接口"""
        logger.info("=" * 50)
        logger.info("测试任务状态查询接口")
        logger.info("=" * 50)
        
        # 测试获取任务列表
        try:
            response = self.session.get(f"{self.base_url}/tasks?limit=10")
            if response.status_code == 200:
                tasks = response.json()
                logger.info(f"获取任务列表成功，共 {len(tasks)} 个任务")
                for task in tasks:
                    logger.info(f"任务: {task.get('task_id')} - 状态: {task.get('status')}")
            else:
                logger.error(f"获取任务列表失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"获取任务列表异常: {str(e)}")
            return False
        
        # 测试获取历史记录
        try:
            response = self.session.get(f"{self.base_url}/history")
            if response.status_code == 200:
                history = response.json()
                logger.info(f"获取历史记录成功: {history}")
            else:
                logger.error(f"获取历史记录失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"获取历史记录异常: {str(e)}")
            return False
        
        return True
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        logger.info("开始运行API测试套件")
        logger.info(f"API地址: {self.base_url}")
        logger.info(f"测试数据路径: {self.test_data_root}")
        logger.info(f"测试输出路径: {self.test_output_dir}")
        
        test_results = {}
        
        # 1. 健康检查
        test_results["health_check"] = self.test_health_check()
        
        # 2. 单图像检测
        test_results["single_image"] = self.test_single_image_detection()
        
        # 3. 单栅格检测
        test_results["single_raster"] = self.test_single_raster_detection()
        
        # 4. 批量图像检测
        test_results["batch_image"] = self.test_batch_image_detection()
        
        # 5. 批量栅格检测
        test_results["batch_raster"] = self.test_batch_raster_detection()
        
        # 6. 等待所有任务完成并查询状态
        logger.info("=" * 50)
        logger.info("等待所有任务完成...")
        logger.info("=" * 50)
        
        for test_name, task_info in self.task_results.items():
            task_id = task_info["task_id"]
            logger.info(f"等待 {test_name} 任务完成: {task_id}")
            
            result = self.wait_for_task_completion(task_id)
            task_info["final_result"] = result
            
            # 检查任务是否成功
            status = result.get("status")
            if status in ["processing_complete", "completed", "success"]:
                test_results[f"{test_name}_execution"] = True
                logger.info(f"{test_name} 任务执行成功")
            else:
                test_results[f"{test_name}_execution"] = False
                logger.error(f"{test_name} 任务执行失败: {result.get('message', '未知错误')}")
        
        # 7. 测试任务状态查询
        test_results["task_status_queries"] = self.test_task_status_queries()
        
        # 8. 生成测试报告
        self.generate_test_report(test_results)
        
        return test_results
    
    def generate_test_report(self, test_results: Dict[str, bool]):
        """生成测试报告"""
        logger.info("=" * 50)
        logger.info("测试报告")
        logger.info("=" * 50)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {failed_tests}")
        logger.info(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        logger.info("\n详细结果:")
        for test_name, result in test_results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"  {test_name}: {status}")
        
        # 保存详细报告到文件
        report_file = self.test_output_dir / "test_report.json"
        report_data = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": passed_tests/total_tests*100
            },
            "detailed_results": test_results,
            "task_results": self.task_results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n详细报告已保存到: {report_file}")
        
        # 检查输出文件
        self.check_output_files()
    
    def check_output_files(self):
        """检查输出文件"""
        logger.info("=" * 50)
        logger.info("检查输出文件")
        logger.info("=" * 50)
        
        output_files = []
        
        # 检查单图像输出
        single_image_files = list(self.test_output_dir.glob("single_image_result_*.png"))
        output_files.extend(single_image_files)
        
        # 检查单栅格输出
        single_raster_files = list(self.test_output_dir.glob("single_raster_result_*.tif"))
        output_files.extend(single_raster_files)
        
        # 检查批量输出目录
        batch_dirs = [
            self.test_output_dir / "batch_image_results",
            self.test_output_dir / "batch_raster_results"
        ]
        
        for batch_dir in batch_dirs:
            if batch_dir.exists():
                files = list(batch_dir.rglob("*"))
                output_files.extend(files)
        
        logger.info(f"找到 {len(output_files)} 个输出文件:")
        for file_path in output_files:
            if file_path.is_file():
                size = file_path.stat().st_size
                logger.info(f"  {file_path} ({size} bytes)")
            else:
                logger.info(f"  {file_path} (目录)")


def main():
    """主函数"""
    # 创建测试器实例
    tester = ChangeDetectionAPITester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 输出最终结果
    logger.info("=" * 50)
    logger.info("测试完成")
    logger.info("=" * 50)
    
    if all(results.values()):
        logger.info("🎉 所有测试通过！")
        return 0
    else:
        logger.error("❌ 部分测试失败，请查看日志了解详情")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
