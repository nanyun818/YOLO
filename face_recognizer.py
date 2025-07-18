import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
from typing import List, Dict, Tuple
import logging

class FaceDetector:
    # YOLO检测相关实现
    pass
class FaceEncoder:
    pass  # 占位符，实际实现需添加人脸识别编码相关方法
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def add_known_face(self, image_path, name):
        """添加已知人脸到识别库"""
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)

    def encode_face(self, face_image):
        """对单张人脸图像进行编码"""
        return face_recognition.face_encodings(face_image)

    def compare_faces(self, face_encoding):
        """比较人脸编码与已知人脸库"""
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return self.known_face_names[best_match_index], face_distances[best_match_index]
        return None, None

class YOLOFaceRecognizer:
    def __init__(self):
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()

    def __init__(self, yolo_model: str = 'yolov8n.pt',
                 face_detector_threshold: float = 0.5,
                 face_recognizer_threshold: float = 0.6):
        """初始化YOLO人脸检测器和InsightFace人脸识别器

        Args:
            yolo_model: YOLO模型名称或路径
            face_detector_threshold: 人脸检测置信度阈值
            face_recognizer_threshold: 人脸识别相似度阈值
        """
        # 加载YOLO模型
        import logging

        # 在类初始化时添加
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 替换print为日志调用
        self.logger.info("人脸识别器初始化完成")
        
        # 添加模型加载错误处理
        try:
            self.yolo_model = YOLO(yolo_model)
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
        self.face_detector_threshold = face_detector_threshold

        self.face_recognizer_threshold = face_recognizer_threshold

        # 存储已知人脸特征库 {name: feature_vector}
        self.known_faces = {}

    def detect_faces(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """使用YOLO检测图像中的人脸

        Args:
            image: BGR格式的图像数组

        Returns:
            人脸区域列表，每个元素为(人脸图像, 边界框坐标(x1,y1,x2,y2))
        """
        # 使用YOLO检测目标，只关注人脸(类别0)
        results = self.yolo_model(image, classes=[0])
        faces = []

        for result in results:
            for box in result.boxes:
                if box.conf[0] >= self.face_detector_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_img = image[y1:y2, x1:x2]
                    faces.append((face_img, (x1, y1, x2, y2)))

        return faces

    def extract_face_feature(self, face_image: np.ndarray) -> np.ndarray:
        """提取人脸特征向量

        Args:
            face_image: 人脸图像数组

        Returns:
            人脸特征向量
        """
        # 转换为RGB格式（face_recognition需要）
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # 提取特征
        encodings = face_recognition.face_encodings(rgb_face)
        if len(encodings) > 0:
            return encodings[0]
        return None

    def add_known_face(self, name: str, face_feature: np.ndarray) -> None:
        """添加已知人脸到特征库

        Args:
            name: 人脸名称
            face_feature: 人脸特征向量
        """
        self.known_faces[name] = face_feature

    def recognize_face(self, face_feature: np.ndarray) -> Tuple[str, float]:
        """识别人脸，返回最相似的已知人脸名称和相似度

        Args:
            face_feature: 待识别人脸特征向量

        Returns:
            (名称, 相似度)，未知人脸返回('Unknown', 0.0)
        """
        if not self.known_faces or face_feature is None:
            return ('Unknown', 0.0)

        max_similarity = 0.0
        recognized_name = 'Unknown'

        # 计算与已知人脸的欧氏距离
        for name, feature in self.known_faces.items():
            # 计算距离，face_recognition返回的是距离，越小越相似
            distance = np.linalg.norm(face_feature - feature)
            # 将距离转换为相似度（0-1范围）
            similarity = 1 / (1 + distance)
            if similarity > max_similarity and similarity >= self.face_recognizer_threshold:
                max_similarity = similarity
                recognized_name = name

        return (recognized_name, max_similarity)

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """处理图像，执行人脸检测和识别

        Args:
            image: BGR格式的图像数组

        Returns:
            (带标注的图像, 识别结果列表)
        """
        results = []
        faces = self.detect_faces(image)

        for face_img, (x1, y1, x2, y2) in faces:
            # 提取人脸特征
            face_feature = self.extract_face_feature(face_img)
            # 识别人脸
            name, similarity = self.recognize_face(face_feature)

            # 绘制边界框和标签
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{name}: {similarity:.2f}'
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            results.append({
                'name': name,
                'similarity': float(similarity),
                'bbox': (x1, y1, x2, y2)
            })

        return image, results

    def process_video(self, video_source, output_path=None):
        # 处理视频文件或摄像头
        # video_source: 可以是视频文件路径或摄像头索引
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {video_source}")

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化视频编写器
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            processed_frame, _ = self.process_image(frame)

            # 显示结果
            cv2.imshow('Face Recognition', processed_frame)

            # 保存结果
            if out:
                out.write(processed_frame)

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 示例用法
    recognizer = YOLOFaceRecognizer()
    print("YOLO人脸识器初始化完成")