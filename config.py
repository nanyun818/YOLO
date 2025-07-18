class Config:
    # YOLO配置
    YOLO_MODEL = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    # 人脸识别配置
    FACE_DISTANCE_THRESHOLD = 0.6
    # 图像处理配置
    RESIZE_WIDTH = 1280
    RESIZE_HEIGHT = 720
    GPU_SUPPORT = True