# YOLO人脸识别系统

基于YOLOv8和PyQt5的实时人脸识别系统，支持摄像头实时检测、人员信息管理和内部人员验证。

## 功能特点
- 实时摄像头人脸识别：支持多摄像头索引尝试，实时捕获并处理视频流
- 人员信息管理：添加、编辑、删除人员信息，支持照片录入
- 内部人员验证：识别内部人员并显示成功提示，未授权人员触发报警
- 数据持久化：使用JSON文件存储人员信息，动态更新

## 环境要求
- Python 3.8+
- PyQt5==5.15.9
- opencv-python==4.8.0.74
- ultralytics==8.0.196
- face-recognition==1.3.0
- numpy==1.24.3

## 安装步骤
1. 克隆仓库：
   ```
   git clone https://github.com/nanyun818/YOLO.git
   cd YOLO
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

3. 运行程序：
   ```
   python main_window.py
   ```

## 使用说明
1. 启动程序后，点击"人员管理"添加内部人员信息
2. 点击"启动摄像头"开始实时人脸识别
3. 系统会自动验证识别结果，内部人员显示成功提示，未授权人员显示错误报警

## 文件结构
- main_window.py: 主窗口UI和逻辑实现
- face_recognizer.py: YOLO人脸识别核心算法
- people_data.json: 人员信息存储文件
- requirements.txt: 项目依赖列表