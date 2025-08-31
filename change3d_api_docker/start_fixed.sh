#!/bin/bash

echo "=========================================="
echo "快速启动API服务（修复版本）"
echo "=========================================="

# 检查是否有现有容器
if docker ps -a | grep -q "change3d-api"; then
    echo "停止现有容器..."
    docker-compose down
fi

# 使用现有镜像，只修改启动命令
echo "启动服务..."
docker run -d \
    --name change3d-api-fixed \
    -p 8000:8000 \
    -v "$(pwd):/app/change3d_api_docker" \
    -v "$(pwd)/../change3d_docker:/app/change3d_docker" \
    -v "$(pwd)/t1:/app/change3d_api_docker/t1" \
    -v "$(pwd)/t2:/app/change3d_api_docker/t2" \
    -v "$(pwd)/output:/app/change3d_api_docker/output" \
    -e PYTHONPATH=/app \
    -e KMP_DUPLICATE_LIB_OK=TRUE \
    change3d-api-optimized:latest \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate change3d_env && cd /app/change3d_api_docker && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo "服务启动完成！"
echo "访问地址: http://localhost:8000/docs"
