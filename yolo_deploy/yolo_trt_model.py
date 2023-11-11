import cv2
import numpy as np
import tensorrt as trt
from tensorrt_test import TRT_model
from trt_model.tensor_calibrator import TensorRT_Calibrator


class YOLO_trt_model(TRT_model):
    def __init__(self, tensorrt_model_path: str, onnx_model_path: str, mode: str = "fp32", trt_int8_calibrator=None):
        super().__init__(tensorrt_model_path, onnx_model_path)
        self.target_dtype = trt.nptype(self.engine.get_tensor_dtype('images'))

    def preProcess(self, image):
        image = cv2.imread(image)
        self.h_d = 640 / image.shape[0]
        self.w_d = 640 / image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(640, 640)).astype(np.float32)
        image = np.transpose(image, (2, 1, 0))
        image = (image / 255)
        return image

    def infer(self, image):
        image = self.preProcess(image)
        # return super().infer(image)
        self.postProcess(image, super().infer(image)[0])

    def postProcess(self, image, data):
        """

        :param image:
        :param data: 1*84*8400
        :return:
        """
        image = cv2.imread("/opt/github_clone/DPLe/bird.jpg")
        data = data[0]
        data = data.T
        pt = data[:, :4]
        classify = data[:, 5:].argmax(1)
        pt_1 = pt[0]
        x, y, w, h = pt_1  # x,y是中心点，w,h是宽高
        x_min, y_min = int(x * 640 / 564), int(y * 640 / 705)
        x_max, y_max = int((x + w) * 640 / 564), int((y + h) * 640 / 705)

        iamge = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        cv2.imshow("a", iamge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_iou(self, box1, box2):
        """

        :param box1: x1,y1,w,h
        :param box2: x2,y2,w,h
        :return:
        """
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2


if __name__ == "__main__":
    yolo = YOLO_trt_model("../trt_model/yolov8x_8.trt", "../onnx_model/yolov8x.onnx",
                          mode="int8", trt_int8_calibrator=TensorRT_Calibrator(input_shape=(1, 3, 224, 224),
                                                                               calibrator_image_dir='../images/cali_images',
                                                                               calibrator_cache_path='../images/cache/int8mode.cache'))
    image = cv2.imread("/opt/github_clone/DPLe/bird.jpg")
    output = yolo.infer("/opt/github_clone/DPLe/bird.jpg")
    print()
