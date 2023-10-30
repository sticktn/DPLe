import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from trt_model.tensor_calibrator import TensorRT_Calibrator


class TRT_model:
    def __init__(self, tensorrt_model_path: str, onnx_model_path: str, logger: trt.ILogger = trt.Logger.INFO,
                 gpu_index: int = 0, mode="fp32", trt_int8_calibrator=None):
        """

        :param trt_model_path: tensorRT model path(.trt)
        :param onnx_model_path: onnx model path (.onnx)
        :param logger: tensorrt logger
        :param gpu_index: index of gpu
        :param mode:Tensor RT model accuracy
        """
        self.onnx_model_path = onnx_model_path
        self.tensorrt_model_path = tensorrt_model_path
        self.output_shape = []
        self.mode = mode.lower()
        self.trt_int8_calibrator = trt_int8_calibrator

        if not os.path.exists(self.tensorrt_model_path):
            self.onnx2tensorrt()

        self.runtime = trt.Runtime(trt.Logger(logger))
        self.engine = self.load_engine(self.tensorrt_model_path)

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

    def onnx2tensorrt(self):
        assert self.mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \
                                                      "but got {}".format(self.mode)
        trt_logger = trt.Logger(getattr(trt.Logger, 'INFO'))
        builder = trt.Builder(trt_logger)

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH) # 创建一个显式批次的网络
        parser = trt.OnnxParser(network, trt_logger) # 将网络和解析器绑定在一起，这样解析器就知道要解析哪个网络了
        with open(self.onnx_model_path, 'rb') as f:
            flag = parser.parse(f.read())
        if not flag:
            for error in range(parser.num_errors):
                print(parser.get_error(error))
        output_tensors = [network.get_output(i) for i in range(network.num_outputs)] # 获取网络中所有的输出张量
        # [network.unmark_output(tensor) for tensor in output_tensors]
        for tensor in output_tensors:
            identity_out_tensor = network.add_identity(tensor).get_output(0)
            identity_out_tensor.name = 'identity_{}'.format(tensor.name)
            network.mark_output(tensor=identity_out_tensor) #将identity_out_tensor标记为网络的输出张量
        config: trt.IBuilderConfig = builder.create_builder_config()
        # config.set_memory_pool_limit(1 << 25)
        if self.mode == 'fp16':
            assert builder.platform_has_fast_fp16, "not support fp16"
            config.set_flag(trt.BuilderFlag.FP16)
        if self.mode == 'int8':
            assert builder.platform_has_fast_int8, "not support int8"
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = self.trt_int8_calibrator
            print(self.trt_int8_calibrator)
        engine = builder.build_serialized_network(network, config)
        with open(self.tensorrt_model_path, 'wb') as f:
            f.write(engine)

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
        """
        infer
        :param input_data: tensor
        :return: output tensor
        """
        stream = cuda.Stream()
        for i in self.d_input:
            cuda.memcpy_htod_async(i, input_data, stream)  # input data to cuda memory

        self.context.execute_async_v2(self.bindings, stream.handle, None)

        for i, j in zip(self.output_data, self.d_output):
            cuda.memcpy_dtoh_async(i, j, stream)

        stream.synchronize()
        # print(self.output_data)
        return self.output_data


if __name__ == '__main__':
    a = TRT_model(tensorrt_model_path='./trt_model/a2_int8.trt',
                  onnx_model_path='./onnx_model/two_label_classify_batch1.onnx',
                  mode='int8',
                  trt_int8_calibrator=TensorRT_Calibrator(input_shape=(1, 3, 224, 224),
                                                          calibrator_image_dir='./images/cali_images',
                                                          calibrator_cache_path='./images/cache/int8mode.cache'))
    # a = two_label_model_trt('trt_model/a1.trt')
    print(a.infer(np.random.randn(1, 3, 224, 224).astype(np.float32)))
