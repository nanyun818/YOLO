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

### 前提条件
- Python 3.8+ 
- 安装CUDA（可选，用于GPU加速）

### 安装步骤
1. 克隆或下载本项目到本地
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 快速开始

#### 1. 图像人脸识别
```python
import cv2
from face_recognizer import YOLOFaceRecognizer

# 初始化识别器
recognizer = YOLOFaceRecognizer()

# 读取图像
image = cv2.imread('test_image.jpg')

# 处理图像
processed_image, results = recognizer.process_image(image)

# 显示结果
cv2.imshow('Face Recognition', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 2. 添加已知人脸
```python
# 读取已知人脸图像
known_face_image = cv2.imread('known_face.jpg')

# 检测并提取人脸特征
faces = recognizer.detect_faces(known_face_image)
if faces:
    face_feature = recognizer.extract_face_feature(faces[0][0])
    recognizer.add_known_face("目标人物", face_feature)
```

#### 3. 视频人脸识别
```python
# 处理视频文件
recognizer.process_video(
    video_path='input_video.mp4',
    output_path='output_video.mp4'
)
```

### 详细API说明

#### YOLOFaceRecognizer类

**初始化参数**:
- `yolo_model`: YOLO模型名称或路径，默认'yolov8n.pt'
- `face_detector_threshold`: 人脸检测置信度阈值，默认0.5
- `face_recognizer_threshold`: 人脸识别相似度阈值，默认0.6

**主要方法**:
- `detect_faces(image)`: 检测图像中的人脸
- `extract_face_feature(face_image)`: 提取人脸特征向量
- `add_known_face(name, face_feature)`: 添加已知人脸到特征库
- `recognize_face(face_feature)`: 识别人脸并返回最相似的已知人脸
- `process_image(image)`: 处理图像，执行人脸检测和识别
- `process_video(video_path, output_path)`: 处理视频文件

## 整合到您的系统

### 模块结构
```
yolo人脸识别/
├── face_recognizer.py  # 核心人脸识别模块
├── requirements.txt    # 依赖包列表
├── example_usage.py    # 使用示例
└── README.md           # 项目文档
```

### 整合步骤
1. 将`face_recognizer.py`添加到您的项目目录
2. 安装所需依赖（参考requirements.txt）
3. 在您的代码中导入并使用YOLOFaceRecognizer类

### 示例整合代码
```python
# 您系统中的代码
from face_recognizer import YOLOFaceRecognizer

class YourSystem:
    def __init__(self):
        # 初始化人脸识别模块
        self.face_recognizer = YOLOFaceRecognizer()
        # 加载您系统中的已知人脸
        self._load_known_faces()

    def _load_known_faces(self):
        # 从您的数据库加载人脸特征
        # 示例: known_faces = db.get_all_known_faces()
        # for name, feature in known_faces.items():
        #     self.face_recognizer.add_known_face(name, feature)
        pass

    def process_frame(self, frame):
        # 处理单帧图像
        processed_frame, results = self.face_recognizer.process_image(frame)
        # 根据识别结果执行您系统的业务逻辑
        for result in results:
            if result['name'] != 'Unknown':
                print(f"识别到: {result['name']}")
                # 执行相应操作
        return processed_frame
```

## 注意事项
- 首次运行时会自动下载YOLO模型和InsightFace模型
- 对于大规模人脸库，建议使用数据库存储人脸特征
- 调整阈值参数可以平衡识别准确率和召回率
- GPU加速可以显著提高处理速度

## 许可证
本项目采用MIT许可证 - 详情参见LICENSE文件