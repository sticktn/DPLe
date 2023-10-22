import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class TRT_model:
    def __init__(self, trt_model_path: str, logger: trt.ILogger = trt.Logger.INFO):
        self.runtime = trt.Runtime(trt.Logger(logger))
        self.engine = self.load_engine(trt_model_path)

        self.context = self.engine.create_execution_context()
        self.input_data = []
        self.d_input = []
        self.d_output = []
        self.output_data = []
        self.bindings = []
        self.target_dtype = trt.nptype(self.engine.get_tensor_dtype('x'))
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            data = np.zeros(self.engine.get_tensor_shape(name), dtype=self.target_dtype)
            d_data = cuda.mem_alloc(data.nbytes)
            self.bindings.append(int(d_data))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_data.append(data)
                self.d_output.append(d_data)
            else:
                self.input_data.append(data)
                self.d_input.append(d_data)

    def load_engine(self, engine_file_path: str) -> trt.ICudaEngine:
        """
        load engine
        :param engine_file_path: path
        :return:
        """
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        stream = cuda.Stream()
        for i in self.d_input:
            cuda.memcpy_htod_async(i, input_data, stream)  # input data to cuda

        self.context.execute_async_v2(self.bindings, stream.handle, None)

        for i, j in zip(self.output_data, self.d_output):
            cuda.memcpy_dtoh_async(i, j, stream)

        stream.synchronize()
        print(self.output_data)
        return self.output_data


class two_label_model_trt(TRT_model):
    def __init__(self, trt_model_path: str):
        super().__init__(trt_model_path)
        self.output_names = ['bird_name_index', 'color_index']

    def run(self,img_batch):
        outputs = self.infer(img_batch)
        return {i: np.argmax(j) for (i, j) in zip(self.output_names, outputs)}



if __name__ == '__main__':
    # infer(engine, input_img, None)
    a = two_label_model_trt('trt_model/a1.trt')
    print(a.run(np.random.randn(1, 3, 224, 224).astype(np.float32)))

