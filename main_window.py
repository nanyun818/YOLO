import sys
import os
import json
print("os模块导入成功")
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QDialog, QFormLayout, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QTimer
import cv2
import numpy as np
from face_recognizer import YOLOFaceRecognizer
import logging

class FaceRecognitionThread(QThread):
    result_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, recognizer, image):
        super().__init__()
        self.recognizer = recognizer
        self.image = image
        
    def run(self):
        # 进行人脸识别处理
        result_image, _ = self.recognizer.process_image(self.image)
        self.result_ready.emit(result_image)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO人脸识别系统")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化人脸识别器
        self.recognizer = YOLOFaceRecognizer()
        
        # 加载系统内部人员数据
        self.internal_people = self.load_internal_people()
        
        # 摄像头相关初始化
        self.cap = None
        self.camera_running = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        
        # 设置UI
        self.init_ui()
        
    def init_ui(self):
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 图像显示标签
        self.image_label = QLabel("请加载图像或启动摄像头")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 人员管理按钮
        self.manage_people_btn = QPushButton("人员管理")
        self.manage_people_btn.clicked.connect(self.open_people_manager)
        button_layout.addWidget(self.manage_people_btn)
        
        # 加载图像按钮
        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_image_btn)
        
        # 启动摄像头按钮
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.clicked.connect(self.start_camera)
        button_layout.addWidget(self.start_camera_btn)
        
        main_layout.addLayout(button_layout)
        
        # 状态栏
        self.statusBar().showMessage("就绪")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg)")
        if file_path:
            # 读取图像
            image = cv2.imread(file_path)
            if image is not None:
                # 启动处理线程
                self.process_thread = FaceRecognitionThread(self.recognizer, image)
                self.process_thread.result_ready.connect(self.update_image)
                self.process_thread.start()
                self.statusBar().showMessage("正在处理图像...")

    def load_internal_people(self):
        """加载系统内部人员数据"""
        if os.path.exists('people_data.json'):
            with open('people_data.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    def update_internal_people(self):
        """更新内部人员数据"""
        self.internal_people = self.load_internal_people()

    def update_image(self, result_image):
        # 转换OpenCV图像到Qt图像
        rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.width(), self.image_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        
        # 验证识别结果是否为内部人员
        self.verify_recognition_results()
        
        self.statusBar().showMessage("处理完成")

    def start_camera(self):
        if not self.camera_running:
            # 尝试打开摄像头（支持多个摄像头索引）
            camera_indices = [0, 1, -1]
            for idx in camera_indices:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    break
            
            if self.cap and self.cap.isOpened():
                self.camera_running = True
                self.timer.start(30)  # 30ms间隔捕获帧
                self.statusBar().showMessage(f"摄像头已启动 (索引: {idx})")
                self.start_camera_btn.setText("停止摄像头")
            else:
                self.statusBar().showMessage("无法打开任何摄像头")
        else:
            # 停止摄像头
            self.stop_camera()

    def stop_camera(self):
        """停止摄像头并释放资源"""
        self.camera_running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_camera_btn.setText("启动摄像头")
        self.statusBar().showMessage("摄像头已停止")

    def update_camera_frame(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                self.stop_camera()
                return
            
            ret, frame = self.cap.read()
            if not ret:
                self.statusBar().showMessage("无法获取摄像头帧")
                return
            
            # 执行人脸识别处理
            processed_frame, results = self.recognizer.process_image(frame)
            # 转换为Qt图像格式
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image).scaled(
                self.image_label.width(), self.image_label.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            
            # 验证识别结果是否为内部人员
            self.verify_recognition_results(results)
        except Exception as e:
            self.logger.error(f"摄像头帧处理错误: {str(e)}")
            self.stop_camera()

    def verify_recognition_results(self, results=None):
        """验证识别结果是否为系统内部人员"""
        # 如果没有传入结果，则从recognizer获取最新结果
        if results is None:
            # 这里假设recognizer有一个last_results属性存储最新识别结果
            results = getattr(self.recognizer, 'last_results', [])
            
        internal_names = [person_data['name'] for person_data in self.internal_people.values()]
        unknown_people = []
        
        for result in results:
            name = result.get('name')
            if name not in internal_names and name != 'Unknown':
                unknown_people.append(name)
        
        if unknown_people:
            # 有未授权人员
            QMessageBox.critical(self, "身份验证错误", 
                                f"检测到未授权人员: {', '.join(unknown_people)}\n请联系管理员获取访问权限")
            self.statusBar().showMessage(f"错误: 检测到未授权人员")
        elif results and all(r['name'] in internal_names for r in results if r['name'] != 'Unknown'):
            # 所有识别到的人员都是内部人员
            authorized_names = [r['name'] for r in results if r['name'] != 'Unknown']
            if authorized_names:
                QMessageBox.information(self, "身份验证成功", 
                                      f"已识别内部人员: {', '.join(authorized_names)}")
                self.statusBar().showMessage(f"已识别内部人员: {', '.join(authorized_names)}")

    def open_people_manager(self):
        dialog = PeopleManagerDialog(self)
        dialog.exec_()
        # 更新内部人员数据
        self.update_internal_people()

class PeopleManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("人员信息管理")
        self.setGeometry(200, 200, 600, 400)
        
        # 初始化人员数据存储
        self.people_data = {}
        self.load_people_data()
        
        # 设置UI
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 表单布局
        form_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.id_input = QLineEdit()
        self.photo_btn = QPushButton("选择照片")
        self.photo_btn.clicked.connect(self.select_photo)
        
        form_layout.addRow("姓名:", self.name_input)
        form_layout.addRow("ID:", self.id_input)
        form_layout.addRow("照片:", self.photo_btn)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加")
        self.add_btn.clicked.connect(self.add_person)
        self.edit_btn = QPushButton("编辑")
        self.edit_btn.clicked.connect(self.edit_person)
        self.delete_btn = QPushButton("删除")
        self.delete_btn.clicked.connect(self.delete_person)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.delete_btn)
        
        # 人员列表
        self.people_table = QTableWidget()
        self.people_table.setColumnCount(3)
        self.people_table.setHorizontalHeaderLabels(["ID", "姓名", "照片路径"])
        self.people_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 加载人员数据到表格
        self.update_people_table()
        
        # 添加到主布局
        layout.addLayout(form_layout)
        layout.addLayout(btn_layout)
        layout.addWidget(self.people_table)
        
        self.setLayout(layout)
        
    def select_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择照片", "", "图像文件 (*.png *.jpg *.jpeg)")
        if file_path:
            self.photo_path = file_path
            self.photo_btn.setText(os.path.basename(file_path))
        
    def add_person(self):
        person_id = self.id_input.text()
        name = self.name_input.text()
        
        if person_id and name and hasattr(self, 'photo_path'):
            self.people_data[person_id] = {
                'name': name,
                'photo_path': self.photo_path
            }
            self.save_people_data()
            self.update_people_table()
            self.clear_form()
        
    def update_people_table(self):
        self.people_table.setRowCount(len(self.people_data))
        for row, (person_id, data) in enumerate(self.people_data.items()):
            self.people_table.setItem(row, 0, QTableWidgetItem(person_id))
            self.people_table.setItem(row, 1, QTableWidgetItem(data['name']))
            self.people_table.setItem(row, 2, QTableWidgetItem(data['photo_path']))
        
    def clear_form(self):
        self.name_input.clear()
        self.id_input.clear()
        self.photo_btn.setText("选择照片")
        if hasattr(self, 'photo_path'):
            delattr(self, 'photo_path')
        
    def load_people_data(self):
        # 从JSON文件加载人员数据
        if os.path.exists('people_data.json'):
            with open('people_data.json', 'r', encoding='utf-8') as f:
                self.people_data = json.load(f)
        
    def save_people_data(self):
        # 保存人员数据到JSON文件
        with open('people_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.people_data, f, ensure_ascii=False, indent=2)

    def edit_person(self):
        selected_row = self.people_table.currentRow()
        if selected_row >= 0:
            person_id = self.people_table.item(selected_row, 0).text()
            person_data = self.people_data.get(person_id)
            if person_data:
                self.id_input.setText(person_id)
                self.name_input.setText(person_data['name'])
                self.photo_path = person_data['photo_path']
                self.photo_btn.setText(os.path.basename(person_data['photo_path']))
                # 在编辑时移除原数据，以便重新添加
                del self.people_data[person_id]
    def delete_person(self):
        selected_row = self.people_table.currentRow()
        if selected_row >= 0:
            person_id = self.people_table.item(selected_row, 0).text()
            if person_id in self.people_data:
                del self.people_data[person_id]
                self.save_people_data()
                self.update_people_table()
        
if __name__ == "__main__":
    # 设置Qt平台插件路径
    env_root = os.path.dirname(sys.executable)
    plugin_path = os.path.join(env_root, "Library", "plugins")
    print(f"设置Qt插件路径: {plugin_path}")
    print(f"路径是否存在: {os.path.exists(plugin_path)}")
    QCoreApplication.addLibraryPath(plugin_path)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {str(e)}")
        sys.exit(1)