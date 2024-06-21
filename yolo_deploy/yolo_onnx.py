import cv2
import numpy as np
import onnxruntime

coco_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                    'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                    'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# coco_class_names = [
#     "safety_helmet",
#     "fuel_tanker",
#     "car",
#     "truck",
#     "other_vehicle",
#     "fire_extinguisher",
#     "fire_blanket",
#     "Calling",
#     "smoking",
#     "smoke",
#     "fire",
#     "person",
# ]


class ONNX_Engine:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.input_names = []
        self.output_names = []
        self.input_shapes = []
        self.output_shapes = []
        self.session = onnxruntime.InferenceSession(self.onnx_model_path)
        input_tensors = self.session.get_inputs()
        for input_tensor in input_tensors:
            self.input_names.append(input_tensor.name)
            self.input_shapes.append(input_tensor.shape)
        output_tensors = self.session.get_outputs()
        for output_tensor in output_tensors:
            self.output_names.append(output_tensor.name)
            self.output_shapes.append(output_tensor.shape)

    def inference(self, input_tensor):
        input_dict = {}
        for _input_tensor, _input_name in zip(input_tensor, self.input_names):
            input_dict[_input_name] = _input_tensor
        outputs = self.session.run(self.output_names, input_dict)
        # for i in np.arange(len(outputs)):
        #     outputs[i] = np.reshape(outputs[i], self.output_shapes[i])
        return outputs


class yolo_onnx_engine(ONNX_Engine):
    def __init__(self, onnx_model_path):
        super().__init__(onnx_model_path)
        self.session.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])
        self.input_W = 640
        self.input_H = 640
        self.h_w = 1080 / 640
        self.w_w = 810 / 640

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, r, top, left

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
        image_resize, self.scale, self.pad_top, self.pad_left = self.letterbox(image)
        # image_resize = cv2.resize(image, dsize=(self.input_W, self.input_H))

        image_resize = image_resize.astype(np.float32)
        image_resize /= 255
        image_resize = np.transpose(image_resize, [2, 0, 1])
        return [np.expand_dims(image_resize, 0)]

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

    def nms(self, input_image, confidence_thres, iou_thres):
        outputs = np.transpose(np.squeeze(input_image[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]
            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)
            # If the maximum score is above the confidence threshold
            if max_score >= confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2))
                top = int((y - h / 2))
                width = int(w)
                height = int(h)
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

        return boxes, scores, class_ids, indices

    def inference(self, image_path):
        input_tensor = self.preprocess(image_path)
        pred = super().inference(input_tensor)
        boxes, scores, class_ids, indices = self.nms(pred, 0.25, 0.7)
        return self.postpocess(boxes, indices, class_ids, scores)

    def postpocess(self, boxes, indices, class_ids, scores):
        for i in indices:
            left, top, width, height = boxes[i]

            # x1 = int(left * self.w_w)
            # y1 = int(top * self.h_w)
            # x2 = int((left + width) * self.w_w)
            # y2 = int((top + height) * self.h_w)

            x1 = int((left - self.pad_left)/self.scale)
            y1 = int((top - self.pad_top)/self.scale)
            x2 = int((left + width-self.pad_left) / self.scale)
            y2 = int((top + height-self.pad_top) / self.scale)
            cv2.rectangle(self.image, (x1, y1),(x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, "{}:{:.2f}".format(coco_class_names[class_ids[i]], scores[i]),
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 155, 255), 2)

        return self.image


if __name__ == "__main__":
    onnx_engine = yolo_onnx_engine("/opt/github_clone/DPLe/onnx_model/jyz.onnx")

    output = onnx_engine.inference("/opt/github_clone/datasets/jyz_fulful/images/test/smoke_fire_57.jpg")
    # cv2.imwrite("output/aaa.jpg", output)
    cv2.imshow("o", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()
