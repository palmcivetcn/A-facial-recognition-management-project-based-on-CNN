# -*- coding: utf-8 -*-
import gc
import os
import time
import cv2
import imutils
import keras
import keras.backend as K
import numpy as np
import psutil
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.utils.generic_utils import CustomObjectScope

# 全局配置参数
EMOTION_RECOGNITION_INTERVAL = 5  # 每隔多少帧进行一次情绪识别
MEMORY_CHECK_INTERVAL = 30  # 每隔多少帧检查一次内存使用情况，减少检查频率
MAX_FRAMES_WITHOUT_GC = 300  # 最大帧数后强制垃圾回收MEMORY_THRESHOLD = 75  # 内存使用率阈值，超过此值进行垃圾回收
MEMORY_THRESHOLD = 75  # 内存使用率阈值，超过此值进行垃圾回收
MAX_CONTINUOUS_RUNTIME = 60 * 10  # 最长连续运行时间（秒），超过此时间强制执行深度清理

# 添加线程锁，防止多线程预测冲突
import threading

prediction_lock = threading.Lock()

# 创建全局图和会话，确保所有线程使用相同的计算图
global_graph = None
global_session = None


# 自定义relu6函数
def relu6(x):
    return K.relu(x, max_value=6)


# 检查GPU可用性并配置
def setup_gpu():
    """检查GPU可用性并进行相应配置"""
    try:
        global global_graph, global_session

        gpu_available = False
        print("正在检查GPU可用性...")

        # 使用tf.test.is_gpu_available检查GPU
        if hasattr(tf, "test") and hasattr(tf.test, "is_gpu_available"):
            gpu_available = tf.test.is_gpu_available()

        # 使用更新的API检查GPU
        elif hasattr(tf.config, "list_physical_devices"):
            gpus = tf.config.list_physical_devices("GPU")
            gpu_available = len(gpus) > 0
            if gpu_available:
                print(f"检测到GPU设备: {gpus}")
                # 尝试设置内存增长
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("已启用GPU内存动态增长")
                except Exception as e:
                    print(f"设置GPU内存增长失败: {str(e)}")

        # 根据GPU可用性配置环境
        if gpu_available:
            print("发现可用GPU，正在配置...")
            # 使用tf.compat.v1代替弃用的API
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 使用80%的GPU内存
            config.gpu_options.allow_growth = True  # 允许GPU内存增长

            # 创建全局图和会话
            global_graph = tf.compat.v1.get_default_graph()
            global_session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(global_session)

            # 取消环境变量的设置，启用GPU
            if (
                "CUDA_VISIBLE_DEVICES" in os.environ
                and os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
            ):
                del os.environ["CUDA_VISIBLE_DEVICES"]

            print("GPU配置成功！")
            return True, config
        else:
            print("未检测到可用GPU，将使用CPU模式运行。")
            # 禁用GPU，使用CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            # 即使在CPU模式下，也创建全局图和会话以保持一致性
            global_graph = tf.compat.v1.get_default_graph()
            config = tf.compat.v1.ConfigProto()
            global_session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(global_session)

            return False, config

    except Exception as e:
        print(f"GPU检查和配置出错: {str(e)}")
        print("将使用CPU模式运行。")
        # 出错时使用CPU模式
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # 即使出错，也确保有全局图和会话
        try:
            global_graph = tf.compat.v1.get_default_graph()
            config = tf.compat.v1.ConfigProto()
            global_session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(global_session)
        except Exception as graph_error:
            print(f"创建全局图和会话失败: {str(graph_error)}")

        return False, config


# 创建情绪预测器类，封装模型加载和预测功能
class EmotionPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.graph = None
        self.session = None
        self.initialized = False
        self.init_moudle = None

    def load_model(self, session_config=None):
        """加载情绪识别模型"""
        try:
            print("正在加载情绪识别模型...")

            # 创建新的图和会话
            self.graph = tf.compat.v1.Graph()
            if session_config is None:
                session_config = tf.compat.v1.ConfigProto()

            with self.graph.as_default():
                self.session = tf.compat.v1.Session(
                    config=session_config, graph=self.graph
                )
                with self.session.as_default():
                    # 在会话中加载模型
                    with CustomObjectScope(
                        {
                            "relu6": relu6,
                            "DepthwiseConv2D": keras.layers.DepthwiseConv2D,
                        }
                    ):
                        self.model = load_model(
                            self.model_path, custom_objects={"tf": tf}
                        )

                        # 确保模型预测函数已创建
                        if hasattr(self.model, "_make_predict_function"):
                            self.model._make_predict_function()

            print("情绪识别模型加载成功！")
            self.initialized = True
            return True
        except Exception as e:
            print(f"情绪识别模型加载失败: {str(e)}")
            self.initialized = False
            return False

    def predict(self, image):
        """预测情绪

        参数:
            image: 预处理后的人脸图像数据

        返回:
            情绪预测概率分布
        """
        if not self.initialized or self.model is None:
            raise ValueError("模型未初始化")

        with self.graph.as_default():
            with self.session.as_default():
                with prediction_lock:  # 使用线程锁保护预测过程
                    return self.model.predict(image, batch_size=1)[0]

    def close(self):
        """释放资源"""
        if self.session is not None:
            self.session.close()
        self.model = None
        self.initialized = False
        gc.collect()


# 模型路径
detection_model_path = (
    "haarcascade_files/haarcascade_frontalface_default.xml"  # 人脸检测模型路径
)
emotion_model_path = "models/best_model/MUL_KSIZE_MobileNet_v2_best.hdf5"  # 情绪识别模型路径

# 尝试设置GPU
using_gpu, session_config = setup_gpu()

# 清空会话并重置默认图
K.clear_session()
face_detection = cv2.CascadeClassifier(detection_model_path)  # 加载人脸检测模型

# 创建情绪预测器实例
emotion_predictor = EmotionPredictor(emotion_model_path)

# 定义情绪标签
EMOTIONS = ["生气", "厌恶", "害怕", "开心", "伤心", "惊讶", "中性"]

# 心理状态分析映射
MENTAL_STATES = {
    "生气": {"state": "压力状态", "description": "可能处于压力或冲突状态", "risk": "低"},
    "厌恶": {"state": "排斥状态", "description": "对某些刺激产生排斥", "risk": "低"},
    "害怕": {"state": "焦虑状态", "description": "可能存在焦虑情绪", "risk": "中"},
    "开心": {"state": "愉悦状态", "description": "心理状态健康", "risk": "无"},
    "伤心": {"state": "抑郁状态", "description": "可能存在抑郁风险", "risk": "高"},
    "惊讶": {"state": "警觉状态", "description": "对刺激反应敏感", "risk": "低"},
    "中性": {"state": "平静状态", "description": "情绪稳定", "risk": "无"},
}

# 加载一个支持中文的字体文件
font_path = "C:/Windows/Fonts/simhei.ttf"  # 请确保这个路径是正确的，或者替换为你的中文字体路径
font_large = ImageFont.truetype(font_path, 30)  # 较大字体
font = ImageFont.truetype(font_path, 20)  # 调整字体大小
small_font = ImageFont.truetype(font_path, 15)  # 调整字体大小


def cv2_add_chinese_text(img, text, position, color, font_size="normal"):
    """在OpenCV图像上添加中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
    draw = ImageDraw.Draw(img_pil)  # 创建一个绘图对象

    # 根据字体大小选择字体
    if font_size == "large":
        draw.text(position, text, font=font_large, fill=color)
    elif font_size == "small":
        draw.text(position, text, font=small_font, fill=color)
    else:
        draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式


# 更新后的EmotionRecognitionApp类定义
class EmotionRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("情绪识别与心理状态分析")
        self.setGeometry(100, 100, 1000, 650)  # 调整窗口大小

        # 设置窗口样式

        # 主窗口布局
        main_layout = QHBoxLayout()

        # 左侧视频显示区域
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.mousePressEvent = self.on_mouse_click  # 添加鼠标点击事件处理

        # 右侧概率显示区域
        self.prob_label = QLabel()
        self.prob_label.setFixedSize(300, 290)
        self.prob_label.setAlignment(Qt.AlignCenter)

        # 右侧心理状态分析区域
        self.analysis_label = QLabel("心理状态分析结果将在这里显示")
        self.analysis_label.setFixedSize(300, 180)
        self.analysis_label.setAlignment(Qt.AlignTop)
        self.analysis_label.setStyleSheet(
            "background-color: white; padding: 10px; border-radius: 5px;"
        )

        # # 添加重置按钮，清除选中的人脸
        # self.reset_button = QPushButton("重置选择")
        # self.reset_button.clicked.connect(self.reset_face_selection)

        # 将控件添加到布局
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.prob_label)
        right_layout.addWidget(self.analysis_label)
        # right_layout.addWidget(self.reset_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 创建主控件并设置布局
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 初始化变量
        self.frame_count = 0
        self.frames_since_last_gc = 0  # 自上次垃圾回收后的帧数
        self.last_label = ""
        self.last_probabilities = None
        self.last_face_coords = (0, 0, 0, 0)
        self.last_canvas = np.zeros((290, 300, 3), dtype="uint8")
        self.last_analysis = None
        self.fps = 0
        self.last_frame_time = time.time()
        self.detected_faces = []  # 存储检测到的所有人脸坐标
        self.selected_face_index = None  # 当前选中的人脸索引
        self.current_frame = None  # 保存当前帧用于处理

        # 内存管理变量
        self.start_time = time.time()
        self.last_deep_clean_time = time.time()
        self.memory_warnings = 0  # 内存警告计数

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "错误", "未检测到摄像头，请检查连接！")
            # 关闭应用
            QtWidgets.QApplication.quit()
            return

        # 创建定时器用于更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms刷新一次，约33fps

        # 创建周期性内存清理定时器
        self.gc_timer = QTimer()
        self.gc_timer.timeout.connect(self.periodic_clean_memory)
        self.gc_timer.start(60000)  # 每60秒检查一次内存状态

    def reset_face_selection(self):
        """重置人脸选择"""
        self.selected_face_index = None
        # 清除上次的情绪和概率结果
        self.last_label = ""
        self.last_probabilities = None
        # 清除画布
        self.last_canvas = np.zeros((290, 300, 3), dtype="uint8")
        self.last_analysis = np.ones((180, 300, 3), dtype="uint8") * 255
        # 更新显示
        self.prob_label.setText("请选择一个人脸进行分析")
        self.analysis_label.setText("心理状态分析结果将在这里显示")

    def on_mouse_click(self, event):
        """处理鼠标点击事件，选择要识别的人脸"""
        if len(self.detected_faces) == 0:
            return

        # 获取点击位置
        x = event.pos().x()
        y = event.pos().y()

        # 检查点击位置是否在某个人脸框内
        selected = False
        for i, (fx, fy, fw, fh) in enumerate(self.detected_faces):
            if fx <= x <= fx + fw and fy <= y <= fy + fh:
                self.selected_face_index = i
                selected = True
                # 立即处理选中的人脸
                if self.current_frame is not None:
                    self.process_selected_face(self.current_frame.copy())
                break

        # 如果点击在没有人脸的区域，则取消选择
        if not selected:
            self.selected_face_index = None

    def process_selected_face(self, frame):
        """处理选中的人脸"""
        if self.selected_face_index is None or self.selected_face_index >= len(
            self.detected_faces
        ):
            return

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (fX, fY, fW, fH) = self.detected_faces[self.selected_face_index]
            self.last_face_coords = (fX, fY, fW, fH)

            # 提取人脸区域
            roi = gray[fY : fY + fH, fX : fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # # 打印ROI形状用于调试
            # print(f"Shape of ROI before prediction: {roi.shape}")

            # 使用情绪预测器进行预测
            try:
                preds = emotion_predictor.predict(roi)
                self.last_probabilities = preds
                self.last_label = EMOTIONS[preds.argmax()]

                # 更新概率画布和分析结果
                self.update_result_display(frame)
            except Exception as predict_error:
                print(f"情绪预测出错: {str(predict_error)}")
                # 尝试重新加载模型并再次预测
                if "input_1_1:0" in str(predict_error):
                    print("尝试重新加载模型...")
                    emotion_predictor.close()
                    if emotion_predictor.load_model(session_config):
                        try:
                            preds = emotion_predictor.predict(roi)
                            self.last_probabilities = preds
                            self.last_label = EMOTIONS[preds.argmax()]
                            self.update_result_display(frame)
                        except Exception as retry_error:
                            print(f"重试预测失败: {str(retry_error)}")
        except Exception as e:
            print(f"处理选中人脸出错: {str(e)}")
            # 为避免卡死，记录上次的预测结果
            if (
                not hasattr(self, "last_probabilities")
                or self.last_probabilities is None
            ):
                # 如果没有上次的结果，创建一个平均分布
                self.last_probabilities = np.ones(len(EMOTIONS)) / len(EMOTIONS)
                self.last_label = "未知"

    def update_frame(self):
        """更新视频帧并进行情绪识别"""
        try:
            start_time = time.time()  # 记录处理开始时间

            # 计算帧率
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                self.fps = 1.0 / time_diff
            self.last_frame_time = current_time

            ret, frame = self.cap.read()
            if not ret:
                return

            self.frame_count += 1
            self.frames_since_last_gc += 1
            process_this_frame = (
                self.frame_count % EMOTION_RECOGNITION_INTERVAL == 0
            )  # 每隔EMOTION_RECOGNITION_INTERVAL帧处理一次情绪识别

            # 限制处理尺寸，降低计算负担
            frame = imutils.resize(frame, width=640)
            self.current_frame = frame.copy()  # 保存当前帧
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 创建概率显示的画布
            canvas = np.zeros((290, 300, 3), dtype="uint8")
            frameClone = frame.copy()

            # 设置人脸检测的最大处理时间
            face_detection_timeout = 0.5  # 秒
            face_detection_start = time.time()

            # 每帧都检测人脸，但使用超时机制避免卡死
            try:
                # 限制最大检测人脸数量
                faces = face_detection.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    maxSize=(300, 300),  # 限制最大人脸尺寸
                )

                # 检查人脸检测是否超时
                if time.time() - face_detection_start > face_detection_timeout:
                    print(f"警告：人脸检测耗时过长 ({time.time() - face_detection_start:.2f}秒)")

                # 限制处理的人脸数量，如果检测到太多人脸，只保留最大的几个
                if len(faces) > 3:
                    # 按面积大小排序，保留最大的3个人脸
                    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[:3]
                    print(f"检测到{len(faces)}张人脸，仅处理最大的3张")

            except Exception as e:
                print(f"人脸检测出错: {str(e)}")
                faces = []  # 出错时设为空列表

            self.detected_faces = faces  # 保存所有检测到的人脸

            # 如果没有检测到人脸，重置选中的人脸索引
            if len(faces) == 0:
                self.selected_face_index = None

            # 在所有检测到的人脸上绘制矩形框
            for i, (fX, fY, fW, fH) in enumerate(faces):
                face_color = (0, 255, 0)  # 默认绿色
                # 如果是选中的人脸，改为红色
                if (
                    self.selected_face_index is not None
                    and i == self.selected_face_index
                ):
                    face_color = (0, 0, 255)  # 红色
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), face_color, 2)

            # 设置情绪识别的最大处理时间
            emotion_timeout = 1.0  # 秒

            if process_this_frame:
                # 如果有选择的人脸，处理该人脸
                if (
                    self.selected_face_index is not None
                    and self.selected_face_index < len(faces)
                ):
                    emotion_start = time.time()

                    # 使用简单直接的方式处理情绪识别，避免多线程问题
                    try:
                        self.process_selected_face(frame)
                    except Exception as e:
                        print(f"情绪识别出错: {str(e)}")

                    # 检查处理时间
                    emotion_time = time.time() - emotion_start
                    if emotion_time > 0.1:
                        print(f"警告：情绪识别处理耗时 {emotion_time:.2f}秒")
                # 如果没有选择的人脸但检测到了人脸，不自动选择，等待用户点击

            # 如果有上一次的结果，且当前仍有选中的人脸，在选中的人脸上显示情绪标签
            if (
                self.last_label
                and self.selected_face_index is not None
                and self.selected_face_index < len(faces)
            ):
                (fX, fY, fW, fH) = faces[self.selected_face_index]  # 使用当前帧中的人脸坐标
                frameClone = cv2_add_chinese_text(
                    frameClone, self.last_label, (fX, fY - 30), (0, 0, 255), "large"
                )

                # 显示最后一次的概率画布
                canvas = self.last_canvas.copy()

            # 显示真实FPS和操作提示
            fps_text = f"FPS: {self.fps:.2f} | 情绪识别频率: 每{EMOTION_RECOGNITION_INTERVAL}帧"
            frameClone = cv2_add_chinese_text(
                frameClone, fps_text, (5, 25), (0, 255, 0), "normal"
            )
            tip_text = "点击人脸选择识别对象 | 点击空白区域取消选择"
            frameClone = cv2_add_chinese_text(
                frameClone, tip_text, (5, 50), (0, 255, 0), "normal"
            )

            # 显示内存使用情况
            if self.frame_count % MEMORY_CHECK_INTERVAL == 0:
                mem = psutil.virtual_memory()
                memory_text = f"内存使用: {mem.percent}%"
                frameClone = cv2_add_chinese_text(
                    frameClone, memory_text, (5, 75), (0, 255, 0), "normal"
                )

            # 转换到Qt格式并显示
            rgb_frame = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

            # 显示概率条
            rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_canvas.shape
            bytes_per_line = ch * w
            qt_canvas = QImage(
                rgb_canvas.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            self.prob_label.setPixmap(QPixmap.fromImage(qt_canvas))

            # 如果有分析结果，显示分析结果
            if hasattr(self, "last_analysis") and self.last_analysis is not None:
                rgb_analysis = cv2.cvtColor(self.last_analysis, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_analysis.shape
                bytes_per_line = ch * w
                qt_analysis = QImage(
                    rgb_analysis.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                self.analysis_label.setPixmap(QPixmap.fromImage(qt_analysis))

            # 定期垃圾回收和内存检查
            if self.frame_count % MEMORY_CHECK_INTERVAL == 0:
                mem = psutil.virtual_memory()
                if mem.percent > 80:  # 内存使用超过80%
                    print(f"警告：内存使用率高 ({mem.percent}%)，执行垃圾回收")
                    K.clear_session()
                    # 重新加载需要的会话
                    emotion_predictor.load_model(session_config)
                    gc.collect()
                    self.frames_since_last_gc = 0

            # 强制周期性垃圾回收，防止内存泄漏
            if self.frames_since_last_gc >= MAX_FRAMES_WITHOUT_GC:
                gc.collect()
                self.frames_since_last_gc = 0

            # 检查本帧处理是否超时
            frame_processing_time = time.time() - start_time
            if frame_processing_time > 0.1:  # 如果处理时间超过100ms，打印警告
                print(f"警告：帧处理耗时过长 ({frame_processing_time:.2f}秒)")

        except Exception as e:
            print(f"帧处理错误: {str(e)}")
            # 出现错误时执行垃圾回收
            gc.collect()

    def update_result_display(self, frame):
        """更新概率画布和分析结果"""
        try:
            # 创建概率显示的画布
            canvas = np.zeros((290, 300, 3), dtype="uint8")

            # 在画布上绘制情绪概率
            canvas = cv2_add_chinese_text(
                canvas, "情绪概率", (10, 5), (255, 255, 255), "normal"
            )

            for (i, (emotion, prob)) in enumerate(
                zip(EMOTIONS, self.last_probabilities)
            ):
                text = f"{emotion}: {prob * 100:.2f}%"
                w = int(prob * 300)
                cv2.rectangle(
                    canvas,
                    (7, (i * 35) + 45),
                    (w, (i * 35) + 75),
                    (0, 0, 255),
                    -1,
                )
                canvas = cv2_add_chinese_text(
                    canvas, text, (10, (i * 35) + 43), (255, 255, 255), "normal"
                )

            # 心理状态分析
            mental_state = MENTAL_STATES[self.last_label]
            mental_text = (
                f"当前情绪: {self.last_label}\n\n"
                f"心理状态: {mental_state['state']}\n\n"
                f"描述: {mental_state['description']}\n\n"
                f"风险程度: {mental_state['risk']}"
            )

            # 显示心理状态分析
            analysis_canvas = np.ones((180, 300, 3), dtype="uint8") * 255
            analysis_canvas = cv2_add_chinese_text(
                analysis_canvas, "心理状态分析:", (10, 5), (0, 0, 0), "normal"
            )

            lines = mental_text.split("\n")
            y_offset = 35
            for line in lines:
                if line:
                    analysis_canvas = cv2_add_chinese_text(
                        analysis_canvas, line, (10, y_offset), (0, 0, 0), "normal"
                    )
                    y_offset += 25

            self.last_canvas = canvas.copy()
            self.last_analysis = analysis_canvas.copy()
        except Exception as e:
            print(f"更新显示出错: {str(e)}")

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        try:
            print("正在关闭应用并释放资源...")

            # 停止所有定时器
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.stop()
            if hasattr(self, "gc_timer") and self.gc_timer is not None:
                self.gc_timer.stop()

            # 释放摄像头
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()

            # 清理情绪预测器
            global emotion_predictor
            emotion_predictor.close()

            # 清理Keras会话
            K.clear_session()

            # 执行多次垃圾回收
            for _ in range(3):
                gc.collect()

            print("资源释放完成")
            event.accept()
        except Exception as e:
            print(f"关闭窗口时出错: {str(e)}")
            event.accept()

    def periodic_clean_memory(self):
        """定期检查内存状态并清理"""
        try:
            # 检查当前内存使用情况
            mem = psutil.virtual_memory()
            runtime = time.time() - self.start_time
            time_since_last_clean = time.time() - self.last_deep_clean_time

            # 显示状态信息
            status_text = (
                f"内存使用率: {mem.percent}%\n"
                f"运行时间: {runtime / 60:.1f}分钟\n"
                f"距上次深度清理: {time_since_last_clean / 60:.1f}分钟"
            )
            print(status_text)

            # 内存使用过高或运行时间过长，执行深度清理
            if (
                mem.percent > MEMORY_THRESHOLD
                or time_since_last_clean > MAX_CONTINUOUS_RUNTIME
            ):
                print(
                    f"执行自动深度清理: 内存使用={mem.percent}%, 距上次清理={time_since_last_clean / 60:.1f}分钟"
                )
                # 使用线程锁保护清理过程
                with prediction_lock:
                    self.last_deep_clean_time = clean_memory(deep_clean=True)
                self.memory_warnings = 0
            else:
                # 执行普通清理
                clean_memory(deep_clean=False)

        except Exception as e:
            print(f"定期内存清理出错: {str(e)}")


# 资源清理函数
def clean_memory(deep_clean=False):
    """清理内存和资源

    参数:
        deep_clean: 是否进行深度清理，重新加载模型
    """
    global emotion_predictor

    try:
        # 执行垃圾回收
        gc.collect()

        # 清理Keras后端会话
        K.clear_session()

        # 如果是深度清理，重新加载模型
        if deep_clean:
            print("执行深度内存清理...")

            # 关闭当前预测器
            emotion_predictor.close()

            # 强制多次垃圾回收
            for _ in range(3):
                gc.collect()

            # 重新加载模型
            emotion_predictor.load_model(session_config)

            # 记录当前时间为最后清理时间
            return time.time()

        return None
    except Exception as e:
        print(f"内存清理出错: {str(e)}")
        return None


def main():
    global emotion_predictor

    try:
        # 初始化TensorFlow设置
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 减少TensorFlow日志输出

        # 禁用急切执行以提高性能
        if (
            hasattr(tf, "compat")
            and hasattr(tf.compat, "v1")
            and hasattr(tf.compat.v1, "disable_eager_execution")
        ):
            tf.compat.v1.disable_eager_execution()
            print("已禁用TensorFlow急切执行模式")

        # 检查模型文件是否存在
        if not os.path.exists(detection_model_path):
            print(f"错误：人脸检测模型文件不存在: {detection_model_path}")
            return

        if not os.path.exists(emotion_model_path):
            print(f"错误：情绪识别模型文件不存在: {emotion_model_path}")
            return

        # 检查可用内存
        mem = psutil.virtual_memory()
        if mem.available < 1 * 1024 * 1024 * 1024:  # 小于1GB可用内存
            print(f"警告：可用内存不足 ({mem.available / (1024 * 1024 * 1024):.2f}GB)，建议关闭其他程序")
            if mem.available < 500 * 1024 * 1024:  # 小于500MB时发出严重警告
                print("严重警告：内存极度不足，程序可能无法正常运行！")
                return

        # 加载情绪识别模型
        if not emotion_predictor.load_model(session_config):
            print("无法加载情绪识别模型，程序将退出")
            return

        # 尝试启动PyQt应用
        try:
            print("尝试启动PyQt GUI界面...")
            print(f"当前情绪识别频率: 每{EMOTION_RECOGNITION_INTERVAL}帧一次")
            print(f"当前运行模式: {'GPU' if using_gpu else 'CPU'}")
            app = QApplication([])

            # 跳转到EmotionRecognitionApp类定义qt窗口控件
            window = EmotionRecognitionApp()
            window.show()
            app.exec_()
        except Exception as e:
            print(f"GUI启动失败: {str(e)}")
            print("正在尝试使用OpenCV原生窗口...")

            # 回退到OpenCV原生窗口
            run_opencv_version()

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    finally:
        # 最终清理
        emotion_predictor.close()
        K.clear_session()
        gc.collect()
        print("程序已结束")


# 使用OpenCV原生窗口运行情绪识别
def run_opencv_version():
    try:
        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：无法打开摄像头，请检查：")
            print("1. 摄像头是否正确连接")
            print("2. 是否有其他程序占用了摄像头")
            return

        # 创建窗口
        cv2.namedWindow("Face", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Emotions", cv2.WINDOW_NORMAL)

        # 初始化变量
        frame_count = 0
        frames_since_gc = 0
        last_label = ""
        last_probabilities = None
        last_face_coords = (0, 0, 0, 0)
        last_canvas = np.zeros((290, 300, 3), dtype="uint8")
        fps = 0
        last_frame_time = time.time()
        detected_faces = []
        selected_face_index = None

        # 内存管理变量
        start_time = time.time()
        last_deep_clean_time = time.time()
        memory_warnings = 0

        # 定义鼠标点击事件处理函数
        def on_mouse_click(event, x, y, flags, param):
            nonlocal selected_face_index, detected_faces
            if event == cv2.EVENT_LBUTTONDOWN:
                # 检查点击位置是否在某个人脸框内
                selected = False
                if len(detected_faces) > 0:  # 确保检测到了人脸
                    for i, (fx, fy, fw, fh) in enumerate(detected_faces):
                        if fx <= x <= fx + fw and fy <= y <= fy + fh:
                            selected_face_index = i
                            selected = True
                            break
                # 如果点击在没有人脸的区域，取消选择
                if not selected:
                    selected_face_index = None

        # 设置鼠标回调
        cv2.setMouseCallback("Face", on_mouse_click)

        print("按'q'键退出程序...")
        print("按'c'键清理内存...")
        print("点击人脸选择识别对象 | 点击空白区域取消选择")
        print(f"当前情绪识别频率: 每{EMOTION_RECOGNITION_INTERVAL}帧一次")
        print(f"当前运行模式: {'GPU' if using_gpu else 'CPU'}")

        while True:
            start_time_frame = time.time()  # 帧处理开始时间

            # 计算帧率
            current_time = time.time()
            time_diff = current_time - last_frame_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            last_frame_time = current_time

            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧，退出...")
                break

            frame_count += 1
            frames_since_gc += 1
            process_this_frame = (
                frame_count % EMOTION_RECOGNITION_INTERVAL == 0
            )  # 每隔EMOTION_RECOGNITION_INTERVAL帧处理一次

            # 限制处理尺寸，降低计算负担
            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 创建概率显示的画布
            canvas = np.zeros((290, 300, 3), dtype="uint8")
            frameClone = frame.copy()

            # 设置人脸检测的最大处理时间
            face_detection_timeout = 0.5  # 秒
            face_detection_start = time.time()

            # 每一帧都检测人脸，但使用超时机制避免卡死
            try:
                # 限制最大检测人脸数量
                faces = face_detection.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    maxSize=(300, 300),  # 限制最大人脸尺寸
                )

                # 检查人脸检测是否超时
                if time.time() - face_detection_start > face_detection_timeout:
                    print(f"警告：人脸检测耗时过长 ({time.time() - face_detection_start:.2f}秒)")

                # 限制处理的人脸数量，如果检测到太多人脸，只保留最大的几个
                if len(faces) > 3:
                    # 按面积大小排序，保留最大的3个人脸
                    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[:3]
                    print(f"检测到{len(faces)}张人脸，仅处理最大的3张")

            except Exception as e:
                print(f"人脸检测出错: {str(e)}")
                faces = []  # 出错时设为空列表

            detected_faces = faces  # 保存所有检测到的人脸

            # 如果没有检测到人脸，重置选中的人脸索引
            if len(faces) == 0:
                selected_face_index = None

            # 在所有检测到的人脸上绘制矩形框
            for i, (fX, fY, fW, fH) in enumerate(faces):
                face_color = (0, 255, 0)  # 默认绿色
                # 如果是选中的人脸，改为红色
                if selected_face_index is not None and i == selected_face_index:
                    face_color = (0, 0, 255)  # 红色
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), face_color, 2)

            # 设置情绪识别的最大处理时间
            emotion_timeout = 1.0  # 秒

            if (
                process_this_frame
                and selected_face_index is not None
                and selected_face_index < len(faces)
            ):
                try:
                    (fX, fY, fW, fH) = faces[selected_face_index]
                    last_face_coords = (fX, fY, fW, fH)

                    # 计时开始
                    emotion_start = time.time()

                    # 提取人脸区域
                    roi = gray[fY : fY + fH, fX : fX + fW]
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # 设置最大超时
                    if time.time() - emotion_start > emotion_timeout:
                        print(f"警告：ROI提取耗时过长，跳过本次情绪识别")
                        continue

                    # 使用情绪预测器进行预测
                    try:
                        preds = emotion_predictor.predict(roi)
                        last_probabilities = preds
                        last_label = EMOTIONS[preds.argmax()]
                    except Exception as predict_error:
                        print(f"情绪预测出错: {str(predict_error)}")
                        if "input_1_1:0" in str(predict_error):
                            print("尝试重新加载模型...")
                            emotion_predictor.close()
                            if emotion_predictor.load_model(session_config):
                                try:
                                    preds = emotion_predictor.predict(roi)
                                    last_probabilities = preds
                                    last_label = EMOTIONS[preds.argmax()]
                                except Exception as retry_error:
                                    print(f"重试预测失败: {str(retry_error)}")

                    # 检查是否超时
                    if time.time() - emotion_start > emotion_timeout:
                        print(f"警告：情绪预测耗时过长 ({time.time() - emotion_start:.2f}秒)")

                    # 在画布上绘制情绪概率
                    canvas = cv2_add_chinese_text(
                        canvas, "情绪概率", (10, 5), (255, 255, 255), "normal"
                    )

                    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                        text = f"{emotion}: {prob * 100:.2f}%"
                        w = int(prob * 300)
                        cv2.rectangle(
                            canvas,
                            (7, (i * 35) + 45),
                            (w, (i * 35) + 75),
                            (0, 0, 255),
                            -1,
                        )
                        canvas = cv2_add_chinese_text(
                            canvas, text, (10, (i * 35) + 43), (255, 255, 255), "normal"
                        )

                    # 心理状态分析
                    mental_state = MENTAL_STATES[last_label]
                    mental_text = (
                        f"当前情绪: {last_label} | "
                        f"心理状态: {mental_state['state']} | "
                        f"风险: {mental_state['risk']}"
                    )

                    # 绘制情绪标签
                    frameClone = cv2_add_chinese_text(
                        frameClone, last_label, (fX, fY - 30), (0, 0, 255), "large"
                    )

                    # 显示脑状态
                    frameClone = cv2_add_chinese_text(
                        frameClone,
                        mental_text,
                        (10, frame.shape[0] - 20),
                        (0, 255, 0),
                        "normal",
                    )

                    last_canvas = canvas.copy()
                except Exception as e:
                    print(f"处理选中人脸出错: {str(e)}")

            elif (
                last_label
                and selected_face_index is not None
                and selected_face_index < len(faces)
            ):
                # 使用上一次的结果，但仅当当前帧中仍然有选中的人脸
                canvas = last_canvas.copy()
                (fX, fY, fW, fH) = faces[selected_face_index]  # 使用当前帧的人脸位置
                frameClone = cv2_add_chinese_text(
                    frameClone, last_label, (fX, fY - 30), (0, 0, 255), "large"
                )

                # 脑状态分析
                if last_label in MENTAL_STATES:
                    mental_state = MENTAL_STATES[last_label]
                    mental_text = (
                        f"当前情绪: {last_label} | "
                        f"脑状态: {mental_state['state']} | "
                        f"风险: {mental_state['risk']}"
                    )
                    frameClone = cv2_add_chinese_text(
                        frameClone,
                        mental_text,
                        (10, frame.shape[0] - 20),
                        (0, 255, 0),
                        "normal",
                    )

            # 显示真实FPS和操作提示
            fps_text = f"FPS: {fps:.2f} | 情绪识别频率: 每{EMOTION_RECOGNITION_INTERVAL}帧"
            frameClone = cv2_add_chinese_text(
                frameClone, fps_text, (5, 25), (0, 255, 0), "normal"
            )
            tip_text = "点击人脸选择识别对象 | 点击空白区域取消选择 | 按q退出 | 按c清理内存"
            frameClone = cv2_add_chinese_text(
                frameClone, tip_text, (5, 50), (0, 255, 0), "normal"
            )

            # 显示内存使用情况
            if frame_count % MEMORY_CHECK_INTERVAL == 0:
                mem = psutil.virtual_memory()
                memory_text = f"内存使用: {mem.percent}%"
                frameClone = cv2_add_chinese_text(
                    frameClone, memory_text, (5, 75), (0, 255, 0), "normal"
                )

            # 显示画面
            cv2.imshow("Face", frameClone)
            cv2.imshow("Emotions", canvas)

            # 定期垃圾回收和内存检查
            if frame_count % MEMORY_CHECK_INTERVAL == 0:
                mem = psutil.virtual_memory()
                time_since_last_clean = current_time - last_deep_clean_time

                # 检查是否需要深度清理
                if (
                    mem.percent > MEMORY_THRESHOLD
                    or time_since_last_clean > MAX_CONTINUOUS_RUNTIME
                ):
                    print(
                        f"执行自动深度清理: 内存使用={mem.percent}%, 距上次清理={time_since_last_clean / 60:.1f}分钟"
                    )
                    last_deep_clean_time = clean_memory(deep_clean=True) or current_time
                    memory_warnings = 0
                    frames_since_gc = 0
                elif mem.percent > 70:  # 内存使用超过70%
                    print(f"警告：内存使用率高 ({mem.percent}%)，执行普通垃圾回收")
                    gc.collect()
                    frames_since_gc = 0
                    memory_warnings += 1

                    # 如果多次出现内存警告，执行深度清理
                    if memory_warnings >= 3:
                        print("多次内存警告，执行深度清理...")
                        last_deep_clean_time = (
                            clean_memory(deep_clean=True) or current_time
                        )
                        memory_warnings = 0

            # 强制周期性垃圾回收，防止内存泄漏
            if frames_since_gc >= MAX_FRAMES_WITHOUT_GC:
                gc.collect()
                frames_since_gc = 0

            # 检查本帧处理是否超时
            frame_processing_time = time.time() - start_time_frame
            if frame_processing_time > 0.1:  # 如果处理时间超过100ms，打印警告
                print(f"警告：帧处理耗时过长 ({frame_processing_time:.2f}秒)")

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            # 按q退出
            if key == ord("q"):
                break

            # 按c清理内存
            elif key == ord("c"):
                print("手动执行内存清理...")
                last_deep_clean_time = clean_memory(deep_clean=True) or current_time
                memory_warnings = 0
                print("内存清理完成")

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        # 清理内存
        K.clear_session()
        gc.collect()

    except Exception as e:
        print(f"OpenCV版本运行错误: {str(e)}")
        # 出错时也要释放资源
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        K.clear_session()
        gc.collect()


if __name__ == "__main__":
    main()
