# -*- coding: utf-8 -*-
"""
描述: 训练情绪分类模型
"""
import gc
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
# from keras.callbacks import ReduceLROnPlateau, TensorBoard
# from keras.preprocessing.image import ImageDataGenerator
# 修改原来的Keras导入为TensorFlow Keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from load_and_process import *
from mobilenet import *
from mobilenet_v2 import *
from mul_ksize_cnn import *


def use_gpu():
    """配置GPU使用"""
    try:
        # 使用tf.compat.v1代替弃用的API
        # from tensorflow.compat.v1.keras.backend import set_session
        # 使用tensorflow.keras代替
        from tensorflow.keras.backend import set_session

        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)  # 使用第一台GPU
        # 使用tf.compat.v1代替弃用的API
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU使用率为90%
        config.gpu_options.allow_growth = True  # 允许容量增长
        set_session(tf.compat.v1.Session(config=config))
        print("GPU配置成功！")
    except Exception as e:
        print(f"GPU配置失败: {str(e)}")
        print("将使用CPU进行训练")


# use_gpu()  # 如果需要使用GPU训练，取消此行注释

# 训练参数
batch_size = 64
num_epochs = 500
input_shape = (48, 48, 1)
verbose = 1
num_classes = 6
patience = 30

# 模型和日志保存路径
trained_model_name = "MUL_KSIZE_MobileNet_v2"
TensorBoard_logdir_path = os.path.abspath("./models/log/MUL_KSIZE_MobileNet_v2")
base_path = "models/"
trained_models_path = base_path + trained_model_name

# 确保日志目录存在
if not os.path.exists(TensorBoard_logdir_path):
    os.makedirs(TensorBoard_logdir_path, exist_ok=True)

if not os.path.exists(base_path):
    os.makedirs(base_path, exist_ok=True)

# 数据增强器
data_generator = ImageDataGenerator(
    featurewise_center=False,  # 设置为False避免ndim错误
    featurewise_std_normalization=False,  # 设置为False避免ndim错误
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)


def train_model():
    """训练情绪识别模型"""
    try:
        # 创建并编译模型
        model = MUL_KSIZE_MobileNet_v2_best(
            input_shape=input_shape, num_classes=num_classes
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.summary()
        with open('model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print("Model summary saved to model_summary.txt")

        # 回调函数
        logs = TensorBoard(log_dir=TensorBoard_logdir_path)  # 保存模型训练日志
        early_stop = EarlyStopping("val_loss", patience=patience)
        reduce_lr = ReduceLROnPlateau(
            "val_loss", factor=0.1, patience=int(patience / 3), verbose=1
        )

        # 模型检查点
        model_names = trained_models_path + ".{epoch:02d}-{val_accuracy:.2f}.hdf5"
        model_checkpoint = ModelCheckpoint(
            model_names, "val_loss", verbose=1, save_best_only=True
        )

        # 定义回调函数列表
        callbacks = [model_checkpoint, early_stop, reduce_lr, logs]

        # 加载数据集
        try:
            # 尝试加载数据集
            print("正在加载FER2013数据集...")
            faces, emotions = load_fer2013(num_classes=num_classes)
            faces = preprocess_input(faces)

            # 数据集分割
            x_train, x_test, y_train, y_test = train_test_split(
                faces, emotions, test_size=0.2, shuffle=True
            )
            x_PublicTest, x_PrivateTest, y_PublicTest, y_PrivateTest = train_test_split(
                x_test, y_test, test_size=0.5, shuffle=True
            )

            print(f"训练集大小: {x_train.shape[0]}")
            print(f"公共测试集大小: {x_PublicTest.shape[0]}")
            print(f"私有测试集大小: {x_PrivateTest.shape[0]}")

            # 训练模型
            print("开始训练模型...")
            # 直接使用model.fit方法训练模型，不使用数据生成器
            # 这样可以避免'NumpyArrayIterator' object has no attribute 'ndim'错误
            history = model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=(x_PublicTest, y_PublicTest),
            )

            # 输出训练好的模型在测试集上的表现
            PublicTest_score = model.evaluate(x_PublicTest, y_PublicTest)
            print("公共测试集得分:", PublicTest_score[0])
            print("公共测试集准确率:", PublicTest_score[1])

            PrivateTest_score = model.evaluate(x_PrivateTest, y_PrivateTest)
            print("私有测试集得分:", PrivateTest_score[0])
            print("私有测试集准确率:", PrivateTest_score[1])

            # 保存模型
            Model_names = (
                trained_models_path
                + "-"
                + "{0:.4f}".format(PublicTest_score[1])
                + "-"
                + "{0:.4f}".format(PrivateTest_score[1])
                + ".hdf5"
            )
            model.save(Model_names)

            print(f"模型已保存为: {Model_names}")

            # 绘制训练历史
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history["loss"], "b-", label="训练损失")
            plt.plot(history.history["val_loss"], "r-", label="验证损失")
            plt.xlabel("轮次")
            plt.ylabel("损失")
            plt.legend()
            plt.title("训练和验证损失")

            plt.subplot(1, 2, 2)
            plt.plot(history.history["accuracy"], "b-", label="训练准确率")
            plt.plot(history.history["val_accuracy"], "r-", label="验证准确率")
            plt.xlabel("轮次")
            plt.ylabel("准确率")
            plt.legend()
            plt.title("训练和验证准确率")

            # 保存图
            plt.savefig(f"{base_path}/training_history.png")
            plt.close()

            return Model_names

        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            print("请确保fer2013数据集已正确加载，数据集路径为: fer2013/fer2013.csv")
            return None
    except Exception as e:
        print(f"模型创建或编译出错: {str(e)}")
        return None
    finally:
        # 清理内存
        gc.collect()
        K.clear_session()


if __name__ == "__main__":
    train_model()
