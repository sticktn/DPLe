# 1. ONNX转TensorRT模型
```python
assert self.mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \  
                                              "but got {}".format(self.mode)  
trt_logger = trt.Logger(getattr(trt.Logger, 'INFO'))  
builder = trt.Builder(trt_logger)  
  
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  
network = builder.create_network(EXPLICIT_BATCH)  
parser = trt.OnnxParser(network, trt_logger)  
with open(self.onnx_model_path, 'rb') as f:  
    flag = parser.parse(f.read())  
if not flag:  
    for error in range(parser.num_errors):  
        print(parser.get_error(error))  
output_tensors = [network.get_output(i) for i in range(network.num_outputs)]  
# [network.unmark_output(tensor) for tensor in output_tensors]  
for tensor in output_tensors:  
    identity_out_tensor = network.add_identity(tensor).get_output(0)  
    identity_out_tensor.name = 'identity_{}'.format(tensor.name)  
    network.mark_output(tensor=identity_out_tensor)  
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
```
# 2. 反序列化trt模型
```python
def load_engine(self, engine_file_path: str) -> trt.ICudaEngine:  
    """  
    load engine    :param engine_file_path: path    :return:  
    """    assert os.path.exists(engine_file_path)  
    print("Reading engine from file {}".format(engine_file_path))  
    with open(engine_file_path, "rb") as f:  
        return self.runtime.deserialize_cuda_engine(f.read())
```

# 3. 生成缓冲区
```python
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
```
# 4. 推理
~~~python
def infer(self, input_data):  
    stream = cuda.Stream()  
    for i in self.d_input:  
        cuda.memcpy_htod_async(i, input_data, stream)  # input data to cuda  
  
    self.context.execute_async_v2(self.bindings, stream.handle, None)  
  
    for i, j in zip(self.output_data, self.d_output):  
        cuda.memcpy_dtoh_async(i, j, stream)  
  
    stream.synchronize()  
    # print(self.output_data)  
    return self.output_data
~~~
