#!/bin/bash

# 开发环境日志查看脚本

echo "=========================================="
echo "查看开发环境日志"
echo "=========================================="

# 查看实时日志
docker-compose -f docker-compose.dev.yml logs -f
