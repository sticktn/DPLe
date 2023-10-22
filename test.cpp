//
// Created by fwq on 23-9-22.
//

#include <iostream>
#include "onnxruntime_cxx_api.h"
#include "onnxModel.h"

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    const char *model_path = "../onnx_model/two_label_classify.onnx";
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char *> input_node_names(num_input_nodes);
    std::vector<const char *> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims;

    bool dynamic_flag = false;
    onnxModel a(env,session_options,model_path);

    //迭代所有的输入节点
    for (int i = 0; i < num_input_nodes; i++) {
        //输出输入节点的名称
//        char *input_name = session.GetInputNameAllocated(i,allocator);
        std::shared_ptr<char> inputName = std::move(session.GetInputNameAllocated(i, allocator));
        char *input_name = inputName.get();
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // 输出输入节点的类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        //输入节点的打印维度
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        //打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++) {
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
            if (input_node_dims[j] < 1) {
                dynamic_flag = true;
            }
        }

        input_node_dims[0] = 1;
    }
    std::vector<size_t> out_shape(num_output_nodes);
    //打印输出节点信息，方法类似
    for (int i = 0; i < num_output_nodes; i++) {
        std::shared_ptr<char> outputName = std::move(session.GetOutputNameAllocated(i, allocator));
        char *output_name = outputName.get();
        printf("Output: %d name=%s\n", i, output_name);
        output_node_names[i] = output_name;
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);
        auto output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++) {
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
            if (j == 1){
                out_shape.at(i) = output_node_dims[j];
            }
        }
    }
    std::cout << input_node_names[0] << std::endl;
    size_t input_tensor_size = 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float) i / (input_tensor_size + 1);
    OrtValue *input_val = nullptr;
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_tensor_values.data(),
                                                              input_tensor_size,
                                                              input_shape.data(),
                                                              4);
    std::cout << "x" << std::endl;
    input_node_names[0] = u8"x";
    output_node_names[0] = u8"softmax_0.tmp_0";
    output_node_names[1] = u8"softmax_1.tmp_0";

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_node_names.data(),
                                      &input_tensor,
                                      1,
                                      output_node_names.data(),
                                      1);
    a.putInputNames({u8"x"});
    a.putOutputNames({u8"softmax_0.tmp_0",u8"softmax_1.tmp_0"});
    auto b2 = a.run(input_tensor,1,1);

    for (int i = 0; i < num_output_nodes; ++i) {
        size_t shape = out_shape.at(i);
        Ort::Value &p = output_tensors.at(0);
        for (int j = 0; j < shape;j++){
            std::cout<< p.At<float>({0,j}) << " ";
        }
        std::cout << std::endl;
    }



    printf("Number of outputs = %zu\n", output_tensors.size());
//    std::cout << output_tensors[0] << std::endl;

    printf("Done!\n");
    return 0;
}