#!/bin/bash

# 开发环境启动脚本 - 支持代码热更新
# 作者: 系统管理员
# 版本: 1.0

set -e

echo "=========================================="
echo "遥感影像变化检测系统 - 开发环境启动"
echo "支持代码热更新"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker环境
check_docker() {
    log_info "检查Docker环境..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装"
        exit 1
    fi
    
    log_success "Docker环境检查通过"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    mkdir -p t1 t2 output
    chmod 755 t1 t2 output
    log_success "目录创建完成"
}

# 构建开发镜像
build_dev_image() {
    log_info "构建开发环境镜像..."
    
    # 使用开发环境配置构建
    docker-compose -f docker-compose.dev.yml build --no-cache
    
    if [ $? -eq 0 ]; then
        log_success "开发镜像构建成功"
    else
        log_error "开发镜像构建失败"
        exit 1
    fi
}

# 启动开发服务
start_dev_service() {
    log_info "启动开发服务（支持热更新）..."
    
    # 停止可能存在的生产环境容器
    docker-compose down 2>/dev/null || true
    
    # 启动开发环境
    docker-compose -f docker-compose.dev.yml up -d
    
    if [ $? -eq 0 ]; then
        log_success "开发服务启动成功"
    else
        log_error "开发服务启动失败"
        exit 1
    fi
}

# 检查服务状态
check_service() {
    log_info "检查服务状态..."
    sleep 5
    
    if docker-compose -f docker-compose.dev.yml ps | grep -q "Up"; then
        log_success "服务运行正常"
    else
        log_error "服务启动异常"
        docker-compose -f docker-compose.dev.yml logs
        exit 1
    fi
}

# 显示开发环境信息
show_dev_info() {
    echo ""
    echo "=========================================="
    echo "开发环境启动完成！"
    echo "=========================================="
    echo ""
    echo "服务信息:"
    echo "  - API文档: http://localhost:8000/docs"
    echo "  - 健康检查: http://localhost:8000/health"
    echo "  - 根路径: http://localhost:8000/"
    echo ""
    echo "热更新说明:"
    echo "  - 修改代码后，服务会自动重启"
    echo "  - 查看实时日志: ./dev_logs.sh"
    echo "  - 手动重启: ./dev_restart.sh"
    echo ""
    echo "管理命令:"
    echo "  - 查看日志: docker-compose -f docker-compose.dev.yml logs -f"
    echo "  - 停止服务: docker-compose -f docker-compose.dev.yml down"
    echo "  - 重启服务: docker-compose -f docker-compose.dev.yml restart"
    echo ""
}

# 主函数
main() {
    check_docker
    create_directories
    build_dev_image
    start_dev_service
    check_service
    show_dev_info
}

main "$@"
