# 1. 构建开发镜像
docker-compose -f docker-compose.dev.yml build --no-cache

# 2. 启动开发服务
docker-compose -f docker-compose.dev.yml up -d

# 3. 查看日志
docker-compose -f docker-compose.dev.yml logs -f

# 4. 修改代码后重启
docker-compose -f docker-compose.dev.yml restart