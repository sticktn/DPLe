import cv2
import numpy as np
import tensorrt as trt
from tensorrt_test import TRT_model
from trt_model.tensor_calibrator import TensorRT_Calibrator
from yolo_deploy.yolo_onnx import coco_class_names
import time

class YOLO_trt_model(TRT_model):
    def __init__(self, tensorrt_model_path: str, onnx_model_path: str, mode: str = "fp32", trt_int8_calibrator=None):
        super().__init__(tensorrt_model_path, onnx_model_path)
        self.target_dtype = trt.nptype(self.engine.get_tensor_dtype('images'))
        self.input_H = 640
        self.input_W = 640
        self.h_w = 0
        self.w_w = 0

    def preprocess(self, image_path):
        if type(image_path) is str:
            image = cv2.imread(image_path)
        else:
            image = image_path
        self.image = image
        h, w, c = image.shape
        self.h_w = h / self.input_H
        self.w_w = w / self.input_W
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image, dsize=(self.input_W, self.input_H))
        image_resize = image_resize.astype(np.float32)
        image_resize /= 255
        image_resize = np.transpose(image_resize, [2, 0, 1])
        return np.ascontiguousarray(np.expand_dims(image_resize, 0))

    def std_output(self, pred):
        """
        将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
        """
        pred = np.squeeze(pred)  # 因为只是推理，所以没有Batch
        pred = np.transpose(pred, (1, 0))
        pred_class = pred[..., 4:]
        pred_conf = np.max(pred_class, axis=-1)
        pred = np.insert(pred, 4, pred_conf, axis=-1)
        return pred  # 8400 * 85

    def xywh2xyxy(self, x):
        """
        将xywh转换为左上角点和左下角点
        Args:
            box:
        Returns: x1y1x2y2
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2

        return y

    def nms(self, dets, thresh):
        # 边界框的坐标
        dets = dets[dets[:, 4] > 0.3]
        dets = self.xywh2xyxy(dets)
        x1 = dets[:, 0]  # 所有行第一列
        y1 = dets[:, 1]  # 所有行第二列
        x2 = dets[:, 2]  # 所有行第三列
        y2 = dets[:, 3]  # 所有行第四列
        # 计算边界框的面积
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)  # (第四列 - 第二列 + 1) * (第三列 - 第一列 + 1)
        # 执行度，包围盒的信心分数
        cls_index = np.argmax(dets[:, 5:], axis=-1)
        scores = dets[:, 4]  # 所有行第五列

        keep = []  # 保留
        keep_cls = []

        # 按边界框的置信度得分排序   尾部加上[::-1] 倒序的意思 如果没有[::-1] argsort返回的是从小到大的
        index = scores[scores > 0.5].argsort()[::-1]  # 对所有行的第五列进行从大到小排序，返回索引值

        # 迭代边界框
        while index.size > 0:  # 6 > 0,      3 > 0,      2 > 0
            i = index[0]  # every time the first is the biggst, and add it directly每次第一个是最大的，直接加进去
            keep.append(i)  # 保存
            # 计算并集上交点的纵坐标（IOU）
            keep_cls.append(cls_index[i])
            x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap计算重叠点
            y11 = np.maximum(y1[i], y1[index[1:]])  # index[1:] 从下标为1的数开始，直到结束
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            # 计算并集上的相交面积
            w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap重叠权值、宽度
            h = np.maximum(0, y22 - y11 + 1)  # the height of overlap重叠高度
            overlaps = w * h  # 重叠部分、交集

            # IoU：intersection-over-union的本质是搜索局部极大值，抑制非极大值元素。即两个边界框的交集部分除以它们的并集。
            #          重叠部分 / （面积[i] + 面积[索引[1:]] - 重叠部分）
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  # 重叠部分就是交集，iou = 交集 / 并集
            # print("ious", ious)
            #               ious <= 0.7
            idx = np.where(ious <= thresh)[0]  # 判断阈值
            # print("idx", idx)
            index = index[idx + 1]  # because index start from 1 因为下标从1开始
        return dets, keep, keep_cls  # 返回保存的值

    def infer(self, image):
        image = self.preprocess(image)
        pred =  super().infer(image)
        pred = self.std_output(pred)
        pred, keep, keep_cls = self.nms(pred, thresh=0.5)
        return self.postpocess(pred, keep, keep_cls)


    def postpocess(self, pred, keep, keep_cls):

        for i, j in zip(keep, keep_cls):
            x1, y1, x2, y2 = pred[i][:4]
            score = pred[i][4]
            cv2.rectangle(self.image, (int(x1 * self.w_w), int(y1 * self.h_w)),
                          (int(x2 * self.w_w), int(y2 * self.h_w)),
                          (0, 0, 255), 2)
            cv2.putText(self.image, "{}:{:.2f}".format(coco_class_names[j], score),
                        (int(x1 * self.w_w), int(y1 * self.h_w)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 155, 255), 2)
        # cv2.imshow("a", self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.image




if __name__ == "__main__":
    yolo = YOLO_trt_model("../trt_model/yolov8x_32.trt", "../onnx_model/yolov8x.onnx",
                          mode="fp32", )

    image_path = "../bird.jpg"
    output = yolo.infer(image_path)
    cv2.imshow("das",output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()
