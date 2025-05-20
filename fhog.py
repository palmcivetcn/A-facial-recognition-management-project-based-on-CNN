import cv2
import numpy as np


# 计算HOG特征
def getFeatureMaps(image, cell_size, mapp):
    if image.ndim == 3 and image.shape[2] == 3:
        # 彩色图像
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image

    h, w = img.shape
    mapp["sizeX"] = int(w / cell_size)
    mapp["sizeY"] = int(h / cell_size)
    mapp["numFeatures"] = 31

    # 计算HOG特征
    winSize = (w, h)
    blockSize = (2 * cell_size, 2 * cell_size)
    blockStride = (cell_size, cell_size)
    cellSize = (cell_size, cell_size)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    feature_map = hog.compute(img)

    # 重塑特征为需要的格式
    feature_map = feature_map.reshape(mapp["sizeY"], mapp["sizeX"], mapp["numFeatures"])
    mapp["map"] = feature_map

    return mapp


# 标准化和截断HOG特征
def normalizeAndTruncate(mapp, alfa):
    # 标准化
    for i in range(mapp["sizeY"]):
        for j in range(mapp["sizeX"]):
            norm = np.sqrt(np.sum(mapp["map"][i, j, :] ** 2)) + 1e-10
            mapp["map"][i, j, :] = mapp["map"][i, j, :] / norm

    # 截断
    mapp["map"] = np.minimum(mapp["map"], alfa)

    # 再次标准化
    for i in range(mapp["sizeY"]):
        for j in range(mapp["sizeX"]):
            norm = np.sqrt(np.sum(mapp["map"][i, j, :] ** 2)) + 1e-10
            mapp["map"][i, j, :] = mapp["map"][i, j, :] / norm

    return mapp


# PCA特征映射
def PCAFeatureMaps(mapp):
    # 这是一个简化的PCA特征映射实现
    # 实际的PCA会更复杂，这里只是保持特征维度不变
    return mapp
