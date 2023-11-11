import cv2
import time
from yolo_onnx import yolo_onnx_engine

onnx_engine = yolo_onnx_engine("../onnx_model/yolov8x.onnx")
video_path = "./videoplayback2.mp4"

cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    start = time.time()

    # 读取帧
    ret, frame = cap.read()

    # 模型推理
    frame = onnx_engine.inference(frame)

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
