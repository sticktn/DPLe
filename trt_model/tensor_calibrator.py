"""
    这是定义TensorRT校准类脚本
"""

import os
import sys
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


# cuda.init()

class TensorRT_Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, input_shape, calibrator_image_dir, calibrator_cache_path):
        """
        这是TensorRT模型抽象INT8量化校准类的初始化函数
        Args:
            input_shape: 输入形状
            calibrator_image_dir: 校准图片集文件夹路径
            calibrator_cache_path: 校准表缓存文件路径
        """
        super(TensorRT_Calibrator, self).__init__()
        # 初始化相关参数
        self.batch_size, self.channel, self.height, self.width = input_shape
        self.cache_file = os.path.abspath(calibrator_cache_path)
        self.calibrator_image_dir = os.path.abspath(calibrator_image_dir)

        # 初始化校准图片路径
        self.image_paths = []
        for image_name in os.listdir(self.calibrator_image_dir):
            self.image_paths.append(os.path.join(self.calibrator_image_dir, image_name))

        self.index = 0
        self.length = len(self.image_paths) // self.batch_size
        self.data_size = self.batch_size * self.channel * self.height * self.width * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)
        self.calibration_data = np.zeros((self.batch_size, self.channel, self.height, self.width), dtype=np.float32)

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                img = cv2.imread(self.image_paths[i + self.index * self.batch_size])
                img = self.preprocess(img)
                self.calibration_data[i] = img
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def get_batch(self, names, p_str=None):
        batch = self.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()

    def preprocess(self, img):
        h, w, c = img.shape
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_w = self.width / w
        r_h = self.height / h
        if r_h > r_w:
            tw = self.width
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.height - th) / 2)
            ty2 = self.height - th - ty1
        else:
            tw = int(r_h * w)
            th = self.height
            tx1 = int((self.width - tw) / 2)
            tx2 = self.width - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        return image
