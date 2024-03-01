import cv2
import time
from yolo_deploy.yolo_trt_model import YOLO_trt_model
from yolo_onnx import yolo_onnx_engine

# engine = yolo_onnx_engine("../onnx_model/yolov8x.onnx")
engine = YOLO_trt_model("../trt_model/yolov8x_8.trt", "../onnx_model/yolov8x.onnx",
                          mode="fp32", )
# video_path = "http://192.168.1.103:4747/video"
video_path = "rtsp://8.140.194.250:8554/rtp/61011300490000000001_34020000001320000001"
# video_path = "/opt/temp/video/00000000210000000.mp4"

cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    start = time.time()

    # 读取帧
    ret, frame = cap.read()

    # 模型推理
    frame = engine.infer(frame)

    # 显示结果帧
    cv2.imshow('frame', frame)

    # 计算该帧处理时间
    end = time.time()
    seconds = end - start
    # fps = 1 / (end - start)
    print(f'FPS: {fps:.2f}')
    # 计算延时等待时间
    delay = max(int(1000/fps - seconds * 1000), 1)
    if cv2.waitKey(delay+10) & 0xFF == ord('q'):
        break
