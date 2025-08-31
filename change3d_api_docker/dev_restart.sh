#!/bin/bash

# 开发环境重启脚本

echo "=========================================="
echo "重启开发环境服务"
echo "=========================================="

# 重启服务
docker-compose -f docker-compose.dev.yml restart

echo "服务重启完成！"
echo "查看日志: ./dev_logs.sh"
