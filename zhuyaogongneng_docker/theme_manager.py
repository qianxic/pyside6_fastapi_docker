"""
3. theme_manager.py
主要功能：负责管理应用程序的主题颜色和样式
定义并管理深色和浅色两套主题
提供各种UI组件的样式定义：按钮、标签、分组框、对话框等
实现主题颜色系统，包括背景色、文本色、边框色等
为不同类型的按钮提供不同样式（主要按钮、次要按钮、工具按钮）
统一管理滚动条、选项卡、列表部件等组件的样式
提供样式选择器和动态样式生成功能


"""

class ThemeManager:
    """集中管理应用程序主题样式的类"""
    
    # 深色主题颜色定义
    @staticmethod
    def get_dark_colors():
        return {
            # 基本颜色
            "background": "#202124",
            "background_secondary": "#2c2c2e",
            "text": "white",
            "secondary_text": "#aaaaaa",
            "border": "#444a5a",
            "separator": "#444a5a",
            
            # 按钮颜色
            "button_primary_bg": "#4e7ae2",
            "button_primary_text": "white",
            "button_primary_hover": "#5c89f2",
            "button_primary_pressed": "#3c69d1",
            
            "button_secondary_bg": "#3e3e40",
            "button_secondary_text": "#f7f7f8",
            "button_secondary_border": "#505050",
            "button_secondary_hover": "#4a4a4c",
            "button_secondary_pressed": "#323234",
            
            "button_utility_bg": "#323234",
            "button_utility_text": "#f7f7f8",
            "button_utility_border": "#444444",
            "button_utility_hover": "#3e3e40",
            "button_utility_pressed": "#262628",
            
            # 控件颜色
            "header_bg": "#2c2c2e",
            "header_text": "#f7f7f8",
            "group_border": "#444a5a",
            "group_title_bg": "#202124",
            "text_edit_bg": "#2c2c2e",
            "text_edit_border": "#444a5a",
            
            # 滚动条
            "scrollbar_bg": "#2c2c2e",
            "scrollbar_handle": "#444a5a",
            
            # 对话框按钮
            "dialog_button_bg": "#444a5a",
            "dialog_button_text": "white",
            "dialog_button_hover": "#5d6576",
            "dialog_button_pressed": "#353b4a",
            
            # 提示图标颜色
            "success_icon": "#00C851",
            "info_icon": "#33b5e5",
            "warning_icon": "#FFD700",
            "error_icon": "#ff4444",
            
            # 容器背景色
            "container_bg": "#2c2c2e",
            "container_border": "#444a5a"
        }
    
    # 浅色主题颜色定义
    @staticmethod
    def get_light_colors():
        return {
            # 基本颜色
            "background": "#ffffff",
            "background_secondary": "#f5f5f7",
            "text": "black",
            "secondary_text": "#666666",
            "border": "#e6e6e6",
            "separator": "#e6e6e6",
            
            # 按钮颜色
            "button_primary_bg": "#4e7ae2",
            "button_primary_text": "white",
            "button_primary_hover": "#5c89f2",
            "button_primary_pressed": "#3c69d1",
            
            "button_secondary_bg": "#f0f0f2",
            "button_secondary_text": "#333333",
            "button_secondary_border": "#e6e6e6",
            "button_secondary_hover": "#e6e6e9",
            "button_secondary_pressed": "#d9d9dc",
            
            "button_utility_bg": "#e6e6e9",
            "button_utility_text": "#333333",
            "button_utility_border": "#d9d9dc",
            "button_utility_hover": "#d9d9dc",
            "button_utility_pressed": "#ccccce",
            
            # 控件颜色
            "header_bg": "#f5f5f7",
            "header_text": "#333333",
            "group_border": "#e6e6e6",
            "group_title_bg": "#ffffff",
            "text_edit_bg": "#f5f5f7",
            "text_edit_border": "#e6e6e6",
            
            # 滚动条
            "scrollbar_bg": "#f5f5f7",
            "scrollbar_handle": "#cccccc",
            
            # 对话框按钮
            "dialog_button_bg": "#f0f0f2",
            "dialog_button_text": "#333333",
            "dialog_button_hover": "#e6e6e9",
            "dialog_button_pressed": "#d9d9dc",
            
            # 提示图标颜色
            "success_icon": "#00C851",
            "info_icon": "#33b5e5",
            "warning_icon": "#FFD700",
            "error_icon": "#ff4444",
            
            # 容器背景色
            "container_bg": "#f5f5f7",
            "container_border": "#e6e6e6"
        }
    
    # 获取当前主题的颜色
    @staticmethod
    def get_colors(is_dark_theme):
        return ThemeManager.get_dark_colors() if is_dark_theme else ThemeManager.get_light_colors()
    
    # 获取主应用程序的主题样式表
    @staticmethod
    def get_app_stylesheet(is_dark_theme):
        colors = ThemeManager.get_colors(is_dark_theme)
        
        return f"""
            QMainWindow, QWidget {{
                background-color: {colors["background"]};
                color: {colors["text"]};
            }}
            QMenuBar, QStatusBar {{
                background-color: {colors["background"]};
                color: {colors["text"]};
            }}
            QHeaderView::section {{
                background-color: {colors["background_secondary"]};
                color: {colors["text"]};
                padding: 4px;
                border: 1px solid {colors["border"]};
            }}
            QGroupBox {{
                border: 1px solid {colors["group_border"]};
                border-radius: 3px;
                margin-top: 8px;
                font-weight: bold;
                color: {colors["text"]};
                background-color: {colors["background"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                background-color: {colors["group_title_bg"]};
            }}
            QTextEdit, QTextBrowser {{
                background-color: {colors["text_edit_bg"]};
                color: {colors["text"]};
                border: 1px solid {colors["text_edit_border"]};
                border-radius: 3px;
            }}
            QPushButton {{
                background-color: {colors["button_secondary_bg"]};
                color: {colors["button_secondary_text"]};
                border: 1px solid {colors["button_secondary_border"]};
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {colors["button_secondary_hover"]};
            }}
            QPushButton:pressed {{
                background-color: {colors["button_secondary_pressed"]};
            }}
            QComboBox {{
                background-color: {colors["button_secondary_bg"]};
                color: {colors["button_secondary_text"]};
                border: 1px solid {colors["button_secondary_border"]};
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 80px;
            }}
            QComboBox:hover {{
                background-color: {colors["button_secondary_hover"]};
                border: 1px solid {colors["button_secondary_border"]};
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors["background_secondary"]};
                color: {colors["text"]};
                selection-background-color: {colors["button_primary_bg"]};
                selection-color: {colors["button_primary_text"]};
                border: 1px solid {colors["border"]};
            }}
            
            /* 滚动条样式 */
            QScrollBar:vertical {{
                border: none;
                background-color: {colors["scrollbar_bg"]};
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: {colors["scrollbar_handle"]};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
                height: 0;
                width: 0;
            }}
            
            /* 水平滚动条样式 */
            QScrollBar:horizontal {{
                border: none;
                background-color: {colors["scrollbar_bg"]};
                height: 10px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {colors["scrollbar_handle"]};
                min-width: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                border: none;
                background: none;
                height: 0;
                width: 0;
            }}
        """
    
    # 获取主题样式表的别名方法，与代码保持一致
    @staticmethod 
    def get_theme_style(is_dark_theme):
        return ThemeManager.get_app_stylesheet(is_dark_theme)
    
    # 获取主要按钮（蓝色）样式
    @staticmethod
    def get_primary_button_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        # 始终使用白色文本，无论是深色还是浅色主题
        return f"""
            QPushButton {{
                background-color: {colors["button_primary_bg"]};
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px 10px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {colors["button_primary_hover"]};
            }}
            QPushButton:pressed {{
                background-color: {colors["button_primary_pressed"]};
            }}
        """
    
    # 获取次要按钮样式
    @staticmethod
    def get_secondary_button_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            QPushButton {{
                background-color: {colors["button_secondary_bg"]};
                color: {colors["button_secondary_text"]};
                border-radius: 4px;
                padding: 5px 10px;
                border: 1px solid {colors["button_secondary_border"]};
            }}
            QPushButton:hover {{
                background-color: {colors["button_secondary_hover"]};
                border: 1px solid {colors["button_secondary_border"]};
            }}
            QPushButton:pressed {{
                background-color: {colors["button_secondary_pressed"]};
                border: 1px solid {colors["button_secondary_border"]};
            }}
        """
    
    # 获取工具类按钮样式
    @staticmethod
    def get_utility_button_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            QPushButton {{
                background-color: {colors["button_utility_bg"]};
                color: {colors["button_utility_text"]};
                border-radius: 4px;
                padding: 5px 10px;
                border: 1px solid {colors["button_utility_border"]};
            }}
            QPushButton:hover {{
                background-color: {colors["button_utility_hover"]};
                border: 1px solid {colors["button_utility_border"]};
            }}
            QPushButton:pressed {{
                background-color: {colors["button_utility_pressed"]};
                border: 1px solid {colors["button_utility_border"]};
            }}
        """
    
    # 获取图像显示区域样式
    @staticmethod
    def get_image_label_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            background-color: {colors["background_secondary"]}; 
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            color: {colors["text"]};
            font-size: 12pt;
        """
    
    # 获取日志文本区域样式
    @staticmethod
    def get_log_text_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        # 深色主题固定使用白色文本，浅色主题固定使用黑色文本
        text_color = "white" if is_dark_theme else "black"
        return f"""
            background-color: {colors["background_secondary"]}; 
            color: {text_color};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            font-family: 'Microsoft YaHei UI';
            padding: 2px;
        """
    
    # 获取导航分隔线样式
    @staticmethod
    def get_separator_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"background-color: {colors['separator']};"
    
    # 获取对话框样式
    @staticmethod
    def get_dialog_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            QDialog {{
                background-color: {colors["background"]};
                color: {colors["text"]};
            }}
            QLabel {{
                color: {colors["text"]};
                font-size: 12px;
                font-weight: bold;
            }}
        """
    
    # 获取对话框标签样式
    @staticmethod
    def get_dialog_label_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            font-size: 13px;
            font-weight: bold;
            margin: 0;
            padding: 5px;
            qproperty-alignment: AlignCenter;
            color: {colors["text"]};
        """
    
    # 获取对话框按钮样式
    @staticmethod
    def get_dialog_button_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            QPushButton {{
                background-color: {colors["dialog_button_bg"]};
                color: {colors["dialog_button_text"]};
                border-radius: 4px;
                padding: 6px 10px;
                min-width: 70px;
            }}
            QPushButton:hover {{
                background-color: {colors["dialog_button_hover"]};
            }}
            QPushButton:pressed {{
                background-color: {colors["dialog_button_pressed"]};
            }}
        """
    
    # 获取透明按钮容器样式
    @staticmethod
    def get_transparent_container_style():
        return "background-color: transparent;"
    
    # 获取容器样式
    @staticmethod
    def get_container_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            background-color: {colors["container_bg"]};
            border: 1px solid {colors["container_border"]};
            border-radius: 4px;
        """
    
    # 获取提示图标样式
    @staticmethod
    def get_icon_style(icon_type="info", is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        color = colors["info_icon"]  # 默认为信息图标颜色
        
        if icon_type == "success":
            color = colors["success_icon"]
        elif icon_type == "warning":
            color = colors["warning_icon"]
        elif icon_type == "error":
            color = colors["error_icon"]
            
        return f"font-size: 28px; color: {color};"
        
    # 获取选项卡部件样式
    @staticmethod
    def get_tab_widget_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            QTabWidget::pane {{
                border: 1px solid {colors["border"]};
                background-color: {colors["background"]};
                border-radius: 3px;
            }}
            QTabBar::tab {{
                background-color: {colors["background"]};
                color: {colors["text"]};
                border: 1px solid {colors["border"]};
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }}
            QTabBar::tab:selected {{
                background-color: {colors["button_primary_bg"]};
                color: {colors["button_primary_text"]};
                border-bottom-color: {colors["button_primary_bg"]};
            }}
            QTabBar::tab:!selected {{
                margin-top: 2px;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {colors["button_secondary_hover"]};
            }}
        """
        
    # 获取列表部件样式
    @staticmethod
    def get_list_widget_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            QListWidget {{
                background-color: {colors["background_secondary"]};
                color: {colors["text"]};
                border: 1px solid {colors["border"]};
                border-radius: 4px;
                padding: 2px;
            }}
            QListWidget::item {{
                padding: 4px;
                border-radius: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {colors["button_primary_bg"]};
                color: {colors["button_primary_text"]};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {colors["button_secondary_hover"]};
            }}
        """
        
    # 获取进度条样式
    @staticmethod
    def get_progress_bar_style(is_dark_theme=False):
        colors = ThemeManager.get_colors(is_dark_theme)
        return f"""
            QProgressBar {{
                border: 1px solid {colors["border"]};
                border-radius: 4px;
                background-color: {colors["background_secondary"]};
                color: {colors["text"]};
                text-align: center;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {colors["button_primary_bg"]};
                border-radius: 3px;
            }}
        """ 