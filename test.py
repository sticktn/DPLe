import numpy as np
import onnxruntime
import numpy
import cv2


class onnx_model:
    def __init__(self, onnx_path):
        self.model = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.model.get_inputs()[0].name

    def get_model_img(self, img_path):
        pass

    def run(self, img):
        pass


class two_label_model(onnx_model):
    def __init__(self, onnx_path):
        super().__init__(onnx_path)
        self.output_names = ['bird_name_index', 'color_index']
        self.img_size = (3, 224, 224)

    def get_model_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        return img

    def run(self, img):
        if isinstance(img, str):
            img = self.get_model_img(img)
        else:
            outputs = self.model.run(None, {self.input_name: img})
            return {i: np.argmax(j) for (i, j) in zip(self.output_names, outputs)}


if __name__ == "__main__":
    model = two_label_model('./onnx_model/two_label_classify_batch1.onnx')
    input_ndarray = np.random.randn(1, 3, 224, 224).astype(dtype=np.float32)
    print(model.run(input_ndarray))
