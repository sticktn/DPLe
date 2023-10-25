# 1. 模型导出
 将已经训练好的Pytorch,TensorFlow,Paddlepaddle,等模型文件通过框架的api导出为onnx模型类型。
# 2. 模型推理
1. 通过onnxruntime将onnxmodel(.onnx模型文件)进行读取，得到session。session = onnxruntime.InferenceSession(onnx_model_path)
2. 获得输入张量和输出张量的名字和形状
```python
input_tensor = session.get_inputs()
for input_tensor in input_tensors:
	input_names.append(input_tensor.name)
	input_shapes.append(input_tensor.shape)
output_tensors = session.get_outputs()
for output_tensor in output_tensors:
	output_names.append(output_tensor.name)
	output_shapes.append(output_tensor.shape)
```
3. 进行推理
>[!notice] session.run 的参数
> 第一个参数是输出张量的名字，第二个参数是字典，key-value对应输入张量的名字和输入张量(np.ndarray)

