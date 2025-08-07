# 项目记录

## 当前项目背景
本项目是一个基于 PySide6 开发的桌面应用程序，旨在提供遥感影像处理相关的功能。它包含一个主应用程序模块 (`zhuyaogongneng_docker.app`)，并可能与一个名为 `change3d_api_docker` 的组件进行交互，用于处理3D变化检测相关的数据。

## 期望实现功能
目前期望实现的功能主要围绕 `start_app.py` 的启动逻辑，包括：
- 成功启动图形用户界面。
- 在后台执行初始化任务，如清理和设置共享目录。
- 确保主应用程序 (`RemoteSensingApp`) 能够正确加载和显示。

## 当前已实现功能
- **应用程序框架**：搭建了基于 PySide6 的应用骨架。
- **后台任务执行器**：实现了 `Worker` 类用于异步执行任务。
- **共享目录管理**：实现了 `setup_shared_directories` 函数，用于在程序启动时创建和清理 `change3d_api_docker` 下的 `t1`, `t2`, `output` 目录。
- **动态模块加载**：通过修改 `sys.path` 确保项目内模块的正确导入。
- **GUI与后台任务通信**：通过信号和槽机制，实现了后台任务状态更新到GUI的逻辑。
- **变化检测功能 (通过外部API)**：
    - **栅格影像处理** (`zhuyaogongneng_docker/function/raster/`):
        - 单个栅格影像变化检测 (`detection.py`): 提供UI接口，调用API执行检测，并在界面上显示结果。
        - 批量栅格影像变化检测 (`batch_processor.py`): 提供UI对话框，允许用户选择多组前后时相栅格目录和输出目录，调用API进行批量处理，并显示日志。
        - 栅格影像渔网分割 (`batch_processor.py`): 在批量处理对话框中提供功能，允许用户对指定目录的栅格影像进行网格化裁剪，使用GDAL进行处理。
    - **普通光学影像处理** (`zhuyaogongneng_docker/function/`):
        - 单个光学影像变化检测 (`change_cd.py`): 提供UI接口，调用API执行检测，并在界面上显示结果。
        - 批量光学影像变化检测 (`batch_processing.py`): 提供UI对话框，允许用户选择多组前后时相影像目录和输出目录，调用API进行批量处理，并显示日志。
        - 光学影像渔网分割 (`batch_processing.py`): 在批量处理对话框中提供功能，允许用户对指定目录的影像进行网格化裁剪，使用OpenCV进行处理。
- **异步任务执行**：广泛使用 `QThread`、`threading.Thread` 和 `concurrent.futures.ThreadPoolExecutor` 来执行耗时操作（API调用、文件处理、图像分割），防止UI阻塞。
- **模块化设计**：功能被拆分到不同的模块和类中，例如 `RasterBatchProcessor`、`RasterChangeDetection`、`BatchProcessingDialog`、`ExecuteChangeDetectionTask` 等。
- **外部API客户端**：通过 `zhuyaogongneng_docker.function.detection_client` 与后端变化检测服务进行通信。
- **UI主题管理**：批量处理对话框等UI组件能够响应主题变化 (`ThemeManager`)。

- **关键变量与交互逻辑**：
    - **`navigation_functions` 对象**: 作为核心UI与功能模块间的桥梁，在初始化时被各主要功能类（如`RasterBatchProcessor`, `RasterChangeDetection`, `BatchProcessingDialog`, `ExecuteChangeDetectionTask`）接收。模块通过它访问主应用的UI元素（如`label_result`）、日志方法(`log_message`)、文件路径(`file_path`, `file_path_after`)和主窗口引用，实现功能模块对主程序状态的依赖注入和UI操作。
    - **输入/输出路径变量**: 用户通过UI选择的目录路径（如`before_dir`, `after_dir`, `output_dir`等）存储在各处理类的实例变量中。这些路径经过处理后，用于构建API请求，定义了任务的数据源和目标。
    - **影像文件列表 (`before_images`, `after_images`, `grid_images`)**: 在批处理和渔网分割中，通过扫描指定目录生成，存储待处理影像的绝对路径列表，为批量操作提供清单，并在任务执行阶段被迭代。
    - **API请求数据 (`data` 字典)**: 在启动检测时构建，封装调用外部`detect_changes` API所需的参数（如`mode`, `before_path`, `after_path`, `output_path`）。此字典传递给后台Worker，规范客户端与服务端的通信。
    - **`Worker` 线程与 `Signals`**: 后台任务（API调用、文件处理）由`Worker`类（通常为QObject，运行在QThread或threading.Thread中）执行。`Worker`通过`Signals`对象（如`WorkerSignals`, `GridSignals`）与主UI线程通信，发射如`finished`（携带结果）、`progress`、`error`、`log`等信号。UI类通过连接槽函数响应这些信号，实现异步处理和状态反馈。
    - **结果数据与显示 (`task_result`, `display_image_path`, `result_image`)**: API调用返回`task_result`字典，其中`display_image_path`指示结果图像位置。客户端读取此路径图像（栅格用GDAL，光学图像用OpenCV）存为NumPy数组（如`result_image`），最终转换为`QPixmap`在UI上显示。 