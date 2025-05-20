# -*- coding: utf-8 -*-
"""
将图片数据集（如CK+的部分数据）处理成FER2013 CSV格式的脚本
"""
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 定义情绪标签映射
# FER2013标准标签: 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
# 根据 'other_dataset' 目录下的文件夹名称进行调整
emotion_labels = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Sad": 4,
    "Surprise": 5
    # "Neutral": 6 # 如果有Neutral文件夹，取消注释并添加
}

# 图片参数
IMG_WIDTH = 48
IMG_HEIGHT = 48

# 数据集路径
INPUT_DATA_DIR = "d:\\ALL_ProjectHome\\Python\\SFD1v1.3\\other_dataset"
OUTPUT_CSV_PATH = "d:\\ALL_ProjectHome\\Python\\SFD1v1.3\\fer2013\\fer2013_custom.csv" # 输出新的csv，避免覆盖示例

def create_fer2013_csv_from_images(input_dir, output_csv):
    print(f"开始从 {input_dir} 处理图片...")
    data = {"emotion": [], "pixels": [], "Usage": []}
    
    # 确保输出目录存在
    output_folder = os.path.dirname(output_csv)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建目录: {output_folder}")

    for emotion_name, emotion_label in emotion_labels.items():
        emotion_dir = os.path.join(input_dir, emotion_name)
        if not os.path.isdir(emotion_dir):
            print(f"警告: 目录 {emotion_dir} 不存在，跳过情绪 {emotion_name}")
            continue

        print(f"处理情绪: {emotion_name} (标签: {emotion_label})")
        images_in_emotion = []
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            try:
                # 读取图片为灰度图
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"警告: 无法读取图片 {img_path}，跳过.")
                    continue
                
                # 调整大小
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                # 展平像素值并转换为空格分隔的字符串
                pixels_str = ' '.join(map(str, img_resized.flatten()))
                images_in_emotion.append((emotion_label, pixels_str))

            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {e}")
        
        if not images_in_emotion:
            print(f"情绪 {emotion_name} 目录中没有找到有效图片.")
            continue

        # 为当前情绪的数据集划分Usage
        # 简单划分为 80% Training, 10% PublicTest, 10% PrivateTest
        # 为了保证每个类别都有数据在各个set，我们对每个emotion文件夹下的图片进行划分
        num_images = len(images_in_emotion)
        train_size = int(0.8 * num_images)
        public_test_size = int(0.1 * num_images)
        # private_test_size = num_images - train_size - public_test_size # 剩余的给PrivateTest

        # 打乱顺序以便随机分配
        np.random.shuffle(images_in_emotion)

        for i, (label, pixels) in enumerate(images_in_emotion):
            data["emotion"].append(label)
            data["pixels"].append(pixels)
            if i < train_size:
                data["Usage"].append("Training")
            elif i < train_size + public_test_size:
                data["Usage"].append("PublicTest")
            else:
                data["Usage"].append("PrivateTest")

    if not data["emotion"]:
        print("没有处理任何图片，无法生成CSV文件。请检查输入目录和图片格式。")
        return

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"成功将 {len(df)} 条数据保存到 {output_csv}")
    print("CSV文件列名: ", df.columns.tolist())
    print("前5行数据:")
    print(df.head())

if __name__ == "__main__":
    # 确保fer2013目录存在，如果脚本直接运行，它可能需要创建这个目录
    # 但通常这个目录应该由 download_fer2013.py 创建
    fer2013_dir = os.path.join("d:\\ALL_ProjectHome\\Python\\SFD1v1.3", "fer2013")
    if not os.path.exists(fer2013_dir):
        os.makedirs(fer2013_dir)
        print(f"创建目录: {fer2013_dir}")

    create_fer2013_csv_from_images(INPUT_DATA_DIR, OUTPUT_CSV_PATH)
    print("\n脚本执行完毕。")
    print(f"请检查生成的CSV文件: {OUTPUT_CSV_PATH}")
    print("如果需要，您可以将此CSV文件重命名为 'fer2013.csv' 并替换 'fer2013' 目录下的同名文件，")
    print("或者修改 'load_and_process.py' 中的 'dataset_path' 指向这个新生成的文件。")