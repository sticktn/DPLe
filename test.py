import numpy as np
import onnxruntime

class ONNX_Engine:
    def __init__(self,onnx_model_path):
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

    def inference(self,input_tensor):
        input_dict = {}
        for _input_tensor,_input_name in zip(input_tensor,self.input_names):
            input_dict[_input_name] = _input_tensor
        outputs = self.session.run(self.output_names,input_dict)
        for i in np.arange(len(outputs)):
            outputs[i] = np.reshape(outputs[i], self.output_shapes[i])
        return outputs



if __name__ == "__main__":
    model = ONNX_Engine('./onnx_model/two_label_classify_batch1.onnx')
    input_ndarray = np.random.randn(1, 3, 224, 224).astype(dtype=np.float32)
    print(model.inference([input_ndarray]))
