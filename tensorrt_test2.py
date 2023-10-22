import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# 1. 确定batch size大小，与导出的trt模型保持一致
BATCH_SIZE = 1

# 2. 选择是否采用FP16精度，与导出的trt模型保持一致
target_dtype = np.float32

# 3. 创建Runtime，加载TRT引擎
f = open("./trt_model/a1.trt", "rb")  # 读取trt模型
runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))  # 创建一个Runtime(传入记录器Logger)
engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())  # 从文件中加载trt引擎
context = engine.create_execution_context()  # 创建context

# 4. 分配input和output内存
input_batch = np.random.randn(BATCH_SIZE, 3, 224, 224).astype(target_dtype)
output1 = np.empty([BATCH_SIZE, 25], dtype=target_dtype)
output2 = np.empty([BATCH_SIZE, 200], dtype=target_dtype)

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output1 = cuda.mem_alloc(1 * output1.nbytes)
d_output2 = cuda.mem_alloc(1 * output2.nbytes)

bindings = [int(d_input), int(d_output1), int(d_output2)]

stream = cuda.Stream()


# 5. 创建predict函数
def predict(batch):  # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
    # transfer predictions back
    cuda.memcpy_dtoh_async(output1, d_output1, stream)
    cuda.memcpy_dtoh_async(output2, d_output2, stream)
    # syncronize threads
    stream.synchronize()

    return output1,output2


# 6. 调用predict函数进行推理，并记录推理时间
def preprocess_input(input):  # input_batch无法直接传给模型，还需要做一定的预处理
    # 此处可以添加一些其它的预处理操作（如标准化、归一化等）
    # result = torch.from_numpy(input).transpose(0, 2).transpose(1, 2)  # 利用torch中的transpose,使(224,224,3)——>(3,224,224)
    # return np.array(result, dtype=target_dtype)
    return input



print("Warming up...")
pred = predict(input_batch)
print(pred)
print("Done warming up!")

t0 = time.time()
pred = predict(input_batch)
t = time.time() - t0
print("Prediction cost {:.4f}s".format(t))
