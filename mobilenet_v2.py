"""MobileNet v2 models for Keras.
# 参考文献
- [用于分类、检测和分割的倒置残差和线性瓶颈移动网络]
   (https://arxiv.org/abs/1801.04381)
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, BatchNormalization, add, Reshape

# 旧的导入方式
# from keras.applications.mobilenet import relu6, DepthwiseConv2D
# 新的导入方式
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Lambda, Concatenate
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model


def relu6(x):
    return ReLU(6.0)(x)


def _conv_block(inputs, filters, kernel, strides):
    """卷积块
    此函数定义了一个带有BN和relu6的2D卷积操作。
    # 参数
        inputs: 张量，卷积层的输入张量。
        filters: 整数，输出空间的维度。
        kernel: 整数或2个整数的元组/列表，指定2D卷积窗口的宽度和高度。
        strides: 整数或2个整数的元组/列表，
            指定卷积沿宽度和高度的步长。
            可以是单个整数，为所有空间维度指定相同的值。
    # 返回
        输出张量。
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = Conv2D(filters, kernel, padding="same", strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """瓶颈结构
    此函数定义了一个基本的瓶颈结构。
    # 参数
        inputs: 张量，卷积层的输入张量。
        filters: 整数，输出空间的维度。
        kernel: 整数或2个整数的元组/列表，指定2D卷积窗口的宽度和高度。
        t: 整数，扩展因子。
            t始终应用于输入大小。
        s: 整数或2个整数的元组/列表，指定沿宽度和高度的卷积步长。
            可以是单个整数，为所有空间维度指定相同的值。
        r: 布尔值，是否使用残差连接。
    # 返回
        输出张量。
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """倒置残差块
    此函数定义了一个或多个相同层的序列。
    # 参数
        inputs: 张量，卷积层的输入张量。
        filters: 整数，输出空间的维度。
        kernel: 整数或2个整数的元组/列表，指定2D卷积窗口的宽度和高度。
        t: 整数，扩展因子。
            t始终应用于输入大小。
        s: 整数或2个整数的元组/列表，指定沿宽度和高度的卷积步长。
            可以是单个整数，为所有空间维度指定相同的值。
        n: 整数，层重复次数。
    # 返回
        输出张量。
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def MobileNet_v2(input_shape, num_classes):
    """MobileNetv2
    此函数定义了MobileNetv2架构。
    # 参数
        input_shape: 整数或3个整数的元组/列表，
            输入张量的形状。
        num_classes: 整数，类别数量。
    # 返回
        MobileNetv2模型。
    """

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.3, name="Dropout")(x)
    x = Conv2D(num_classes, (1, 1), padding="same")(x)

    x = Activation("softmax", name="softmax")(x)
    output = Reshape((num_classes,))(x)
    model = Model(inputs, output)
    return model


def MobileNet_v2_MIX(input_shape, num_classes):
    """MobileNetv2
    此函数定义了MobileNetv2架构。
    # 参数
        input_shape: 整数或3个整数的元组/列表，
            输入张量的形状。
        num_classes: 整数，类别数量。
    # 返回
        MobileNetv2模型。
    """

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x1 = _inverted_residual_block(x, 16, (5, 5), t=1, strides=1, n=1)
    x = _inverted_residual_block(x1, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x2 = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x2, 96, (5, 5), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x3 = _inverted_residual_block(x, 320, (5, 5), t=6, strides=1, n=1)

    x4 = _conv_block(x3, 1280, (1, 1), strides=(1, 1))

    n = int(x4.shape[1])
    X1 = Lambda(lambda X: tf.compat.v1.image.resize_images(X, size=(n, n)))(x1)
    X2 = Lambda(lambda X: tf.compat.v1.image.resize_images(X, size=(n, n)))(x2)
    # X3 = Lambda(lambda X: tf.image.resize_images(X, size=(n, n)))(x3)
    x = Concatenate(axis=-1)([X2, X1, x4])

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, int(x.shape[1])))(x)
    x = Dropout(0.3, name="Dropout")(x)
    x = Conv2D(num_classes, (1, 1), padding="same")(x)

    x = Activation("softmax", name="softmax")(x)
    output = Reshape((num_classes,))(x)
    model = Model(inputs, output)
    return model


if __name__ == "__main__":
    input_shape = (48, 48, 1)
    num_classes = 7
    model = MobileNet_v2(input_shape, num_classes)
    # plot_model(model, 'models/MobileNet_v2.png', show_shapes=True, show_layer_names=True)  # 保存模型图
    model.summary()
