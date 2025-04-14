# 使用PyTorch官方CUDA+Conda基础镜像
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    ffmpeg \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建目录结构 (调整，确保目标目录存在)
RUN mkdir -p /app/change3d_api_docker /app/change3d_docker \
    /app/change3d_api_docker/t1 /app/change3d_api_docker/t2 /app/change3d_api_docker/output

# 复制 requirements.txt (路径相对于新的构建上下文)
COPY ./change3d_api_docker/requirements.txt /app/change3d_api_docker/

# 设置conda环境变量
ENV PATH /opt/conda/bin:$PATH

# <<< 添加：更新 Conda 自身 >>>
RUN conda update -n base -c defaults conda -y --solver=libmamba && conda clean -afy

# <<< 添加：创建并激活新环境 >>>
RUN conda create -n change3d_env python=3.10 -y && echo "conda activate change3d_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/change3d_env/bin:$PATH
SHELL ["conda", "run", "-n", "change3d_env", "/bin/bash", "-c"]

# 在新环境中使用conda安装复杂的地理空间依赖
RUN conda install -y -c conda-forge \
    gdal \
    geopandas \
    rasterio \
    fiona \
    shapely \
    pyproj \
    fastapi \
    uvicorn \
    && conda clean -afy

# 使用pip安装其他依赖 (现在会在 change3d_env 环境中安装)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install fastapi uvicorn python-multipart aiofiles pytorchvideo

# 过滤并安装其他Python依赖 (现在会在 change3d_env 环境中安装)
RUN grep -v "pywin32" /app/change3d_api_docker/requirements.txt | grep -v "torch" | grep -v "gdal" | grep -v "rasterio" | grep -v "geopandas" | grep -v "pyproj" | grep -v "fiona" | grep -v "shapely" > /tmp/requirements_filtered.txt && \
    pip install -r /tmp/requirements_filtered.txt || echo "部分依赖安装失败，但继续构建"

# 创建.project_root文件
RUN touch /app/.project_root

# 复制项目文件到容器 (调整路径)
COPY ./change3d_api_docker/ /app/change3d_api_docker/
# <<< 添加复制 change3d_docker >>>
COPY ./change3d_docker/ /app/change3d_docker/

# 添加 PyTorch 安全补丁文件
RUN echo 'import torch.serialization' > /app/patch_pytorch.py && \
    echo 'import numpy' >> /app/patch_pytorch.py && \
    echo 'print("应用 PyTorch 安全补丁...")' >> /app/patch_pytorch.py && \
    echo 'safe_globals = [' >> /app/patch_pytorch.py && \
    echo '    numpy.core.multiarray.scalar,' >> /app/patch_pytorch.py && \
    echo '    numpy.core.multiarray._reconstruct,' >> /app/patch_pytorch.py && \
    echo '    numpy.ndarray' >> /app/patch_pytorch.py && \
    echo ']' >> /app/patch_pytorch.py && \
    echo 'torch.serialization.add_safe_globals(safe_globals)' >> /app/patch_pytorch.py && \
    echo 'print("PyTorch 安全补丁已应用。")' >> /app/patch_pytorch.py


# 设置 PyTorch 环境变量 (可选，但建议保留以防万一)
ENV TORCH_LOAD_LEGACY_WARNINGS=0
ENV TORCH_WARN_OF_UNSAFE_SCRIPT_LOADING=0

# 修改 start_api.sh 的创建过程，加入补丁执行
RUN echo '#!/bin/bash' > /app/start_api.sh && \
    echo '# Activate Conda environment' >> /app/start_api.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /app/start_api.sh && \
    echo 'conda activate change3d_env' >> /app/start_api.sh && \
    echo '' >> /app/start_api.sh && \
    echo '# Apply PyTorch safety patch' >> /app/start_api.sh && \
    echo 'python /app/patch_pytorch.py' >> /app/start_api.sh && \
    echo '' >> /app/start_api.sh && \
    echo '# Debug output' >> /app/start_api.sh && \
    echo 'echo "--- Environment Info ---"' >> /app/start_api.sh && \
    echo 'echo "Current directory: $(pwd)"' >> /app/start_api.sh && \
    echo 'echo "PYTHONPATH: $PYTHONPATH"' >> /app/start_api.sh && \
    echo 'echo "Conda env: $CONDA_DEFAULT_ENV"' >> /app/start_api.sh && \
    echo 'echo "Which Python: $(which python)"' >> /app/start_api.sh && \
    echo 'echo "------------------------"' >> /app/start_api.sh && \
    echo '' >> /app/start_api.sh && \
    echo '# <<< 保持之前的调试 ls 命令 >>>' >> /app/start_api.sh && \
    echo 'echo "--- Contents of /app at startup ---"' >> /app/start_api.sh && \
    echo 'ls -la /app' >> /app/start_api.sh && \
    echo 'echo "------------------------------------"' >> /app/start_api.sh && \
    echo '' >> /app/start_api.sh && \
    echo 'echo "启动API服务..."' >> /app/start_api.sh && \
    echo 'exec uvicorn change3d_api_docker.main:app --host 0.0.0.0 --port 8000' >> /app/start_api.sh && \
    chmod +x /app/start_api.sh

ENV PYTHONPATH=/app
ENV KMP_DUPLICATE_LIB_OK=TRUE

# 修复可能的GDAL路径问题
RUN if [ -f /opt/conda/lib/libgdal.so ]; then \
    ln -s /opt/conda/lib/libgdal.so /usr/lib/libgdal.so || true; \
    fi

# 暴露API端口
EXPOSE 8000

# 启动API服务 - 使用脚本
CMD ["/app/start_api.sh"]
#docker-compose build
#docker-compose up -d

# 在Dockerfile中使用weights_only=False
RUN echo '# 修改模型加载代码' > /app/patch_model_load.py && \
    echo 'import torch.serialization' >> /app/patch_model_load.py && \
    echo 'torch.serialization.add_safe_globals(["numpy.core.multiarray.scalar"])' >> /app/patch_model_load.py && \
    echo 'print("模型加载安全设置已应用")' >> /app/patch_model_load.py