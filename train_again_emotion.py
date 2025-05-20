# -*- coding: utf-8 -*-
"""
描述: 使用迁移学习重新训练情绪分类模型
"""
import gc

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Reshape, Dropout, Conv2D, Activation
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import CustomObjectScope
from sklearn.model_selection import train_test_split

from load_retrain_data import *
from mobilenet import *
from mobilenet_v2 import *
from mul_ksize_cnn import *

# 参数设置
batch_size = 16
num_epochs = 500
input_shape = (48, 48, 1)
verbose = 1
num_classes = 6  # 注意：重新训练时可能会修改类别数量
patience = 30

# 数据集和模型路径
retrain_data_path = "other_dataset"  # 新数据集路径
load_model_path = "models/best_model/MUL_KSIZE_MobileNet_v2_best.hdf5"  # 预训练模型路径

# 模型保存路径和名字
base_path = "models/"
running_model_name = "MUL_KSIZE_MobileNet_v2_best_retrained"
trained_models_path = base_path + running_model_name
TensorBoard_logdir_path = "./models/log/MUL_KSIZE_MobileNet_v2_best_retrained"

# 确保目录存在
os.makedirs(os.path.dirname(trained_models_path), exist_ok=True)
os.makedirs(TensorBoard_logdir_path, exist_ok=True)

# 数据增强器
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)


def retrain_model():
    """重新训练情绪识别模型"""
    global num_classes
    try:
        # 检查预训练模型是否存在
        if not os.path.isfile(load_model_path):
            print(f"错误：预训练模型不存在: {load_model_path}")
            return None

        print(f"正在加载预训练模型: {load_model_path}")
        try:
            # 加载预训练模型
            with CustomObjectScope(
                {
                    "relu6": relu6,
                    "DepthwiseConv2D": keras.layers.DepthwiseConv2D,
                }
            ):
                pre_model = load_model(load_model_path, custom_objects={"tf": tf})
                print("预训练模型加载成功")
        except Exception as e:
            print(f"加载预训练模型失败: {str(e)}")
            print("尝试从头创建模型")
            pre_model = MUL_KSIZE_MobileNet_v2_best(
                input_shape=input_shape, num_classes=7
            )

        # 重建网络以适应新的类别数量
        try:
            # 提取预训练模型的特征提取层
            pooling_output = pre_model.get_layer("global_average_pooling2d_1").output
            x = Reshape((1, 1, 1280))(pooling_output)
            x = Dropout(0.3, name="Dropout")(x)
            x = Conv2D(num_classes, (1, 1), padding="same", name="conv2d_retrained")(x)
            x = Activation("softmax", name="softmax")(x)
            output = Reshape((num_classes,))(x)
            model = Model(
                inputs=pre_model.input, outputs=output, name="mobilenetv2_FER_retrained"
            )

            # 编译模型
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            model.summary()
            print("模型重建成功")
        except Exception as e:
            print(f"模型重建失败: {str(e)}")
            print("使用默认模型构建")
            model = MUL_KSIZE_MobileNet_v2_best(
                input_shape=input_shape, num_classes=num_classes
            )
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            model.summary()

        # 回调函数
        logs = TensorBoard(log_dir=TensorBoard_logdir_path)
        early_stop = EarlyStopping("val_loss", patience=patience)
        reduce_lr = ReduceLROnPlateau(
            "val_loss", factor=0.1, patience=int(patience / 3), verbose=1
        )

        # 模型检查点
        model_names = trained_models_path + ".{epoch:02d}-{val_accuracy:.2f}.hdf5"
        model_checkpoint = ModelCheckpoint(
            model_names, "val_loss", verbose=1, save_best_only=True
        )

        callbacks = [model_checkpoint, early_stop, reduce_lr, logs]

        # 加载重新训练的数据集
        try:
            print(f"正在加载新数据集: {retrain_data_path}")
            faces, emotions = load_retrain_data(
                retrain_data_path,
                file_ext="*.png",
                image_size=(input_shape[1], input_shape[0]),
                channel=input_shape[2],
                crop_face=True,
            )

            if len(faces) == 0:
                print("错误：数据集为空")
                return None

            print(f"加载了 {len(faces)} 张图像")
            print(f"类别数量: {emotions.shape[1]}")

            # 检查类别数量是否匹配
            if emotions.shape[1] != num_classes:
                print(f"警告：类别数量不匹配 (模型: {num_classes}, 数据: {emotions.shape[1]})")
                print("重建模型以匹配数据类别数量...")

                # 重建模型以匹配类别数量
                num_classes = emotions.shape[1]
                pooling_output = pre_model.get_layer(
                    "global_average_pooling2d_1"
                ).output
                x = Reshape((1, 1, 1280))(pooling_output)
                x = Dropout(0.3, name="Dropout")(x)
                x = Conv2D(
                    num_classes, (1, 1), padding="same", name="conv2d_retrained"
                )(x)
                x = Activation("softmax", name="softmax")(x)
                output = Reshape((num_classes,))(x)
                model = Model(
                    inputs=pre_model.input,
                    outputs=output,
                    name="mobilenetv2_FER_retrained",
                )
                model.compile(
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"],
                )

            # 数据集分割
            x_train, x_test, y_train, y_test = train_test_split(
                faces, emotions, test_size=0.1, shuffle=True
            )
            print(f"训练集大小: {x_train.shape[0]}")
            print(f"测试集大小: {x_test.shape[0]}")

            # 训练模型
            print("开始训练模型...")
            history = model.fit(
                data_generator.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                epochs=num_epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=(x_test, y_test),
            )

            # 评估模型
            test_score = model.evaluate(x_test, y_test)
            print("测试集损失:", test_score[0])
            print("测试集准确率:", test_score[1])

            # 保存最终模型
            final_model_path = f"{trained_models_path}-{test_score[1]:.4f}.hdf5"
            model.save(final_model_path)
            print(f"模型已保存为: {final_model_path}")

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
            history_image_path = f"{base_path}/retrain_history.png"
            plt.savefig(history_image_path)
            plt.close()
            print(f"训练历史已保存为: {history_image_path}")

            # 如果训练成功，预测一些测试样本并显示结果
            if len(x_test) > 0:
                predictions = model.predict(x_test[:10])
                # 获取类别标签
                emotion_labels = []
                for i in range(num_classes):
                    for j in range(num_classes):
                        if y_test[0][j] == 1:
                            emotion_labels.append(j)
                            break

                print("\n预测结果示例:")
                for i in range(min(5, len(predictions))):
                    pred_class = np.argmax(predictions[i])
                    true_class = np.argmax(y_test[i])
                    print(
                        f"样本 {i + 1}: 真实类别 = {true_class}, 预测类别 = {pred_class}, 置信度 = {predictions[i][pred_class]:.4f}"
                    )

            return final_model_path

        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            import traceback

            traceback.print_exc()
            return None
    except Exception as e:
        print(f"重新训练过程出错: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        # 清理内存
        gc.collect()
        K.clear_session()


if __name__ == "__main__":
    retrain_model()
