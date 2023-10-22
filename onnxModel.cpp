//
// Created by guo on 23-9-25.
//

#include "onnxModel.h"
#include<cstring>
#include <codecvt>

onnxModel::onnxModel(Ort::Env &env, Ort::SessionOptions &session_options, const char *model_path) {
// TODO 获取input names 和 output names 有问题，会变化
    session = Ort::Session(env, model_path, session_options);
    num_input_nodes = session.GetInputCount();
    num_output_nodes = session.GetOutputCount();
    input_node_names = std::vector<const char *>(num_input_nodes);
    output_node_names = std::vector<const char *>(num_output_nodes);
    input_nodes_dims = std::vector<std::vector<int64_t>>(num_input_nodes);
    output_nodes_dims = std::vector<std::vector<int64_t>>(num_output_nodes);
    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < num_input_nodes; i++) {
        std::shared_ptr<char> inputName = std::move(session.GetInputNameAllocated(i, allocator));
        const char *input_name = inputName.get();
        input_node_names[i] = input_name;
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//        ONNXTensorElementDataType type = tensor_info.GetElementType();
        auto input_node_dims = tensor_info.GetShape();
        input_nodes_dims.at(i) = input_node_dims;
    }
    for (int i = 0; i < num_output_nodes; i++) {
        std::shared_ptr<char> outputName = std::move(session.GetOutputNameAllocated(i, allocator));
        char *output_name = outputName.get();
        output_node_names[i] = output_name;
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//        ONNXTensorElementDataType type = tensor_info.GetElementType();
        auto output_node_dims = tensor_info.GetShape();
        output_nodes_dims.at(i) = output_node_dims;
    }
}

void onnxModel::putInputNames(std::vector<const char *> inputNames) {
    for (int i = 0; i < num_input_nodes; ++i) {
        input_node_names[i] = inputNames[i];
    }
}

void onnxModel::putOutputNames(std::vector<const char *> outputNames) {
    for (int i = 0; i < num_output_nodes; ++i) {
        output_node_names[i] = outputNames[i];
    }
}

Ort::Value onnxModel::run(Ort::Value &input_tensor, size_t inputSize, size_t outputSize) {
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_node_names.data(),
                                      &input_tensor,
                                      inputSize,
                                      output_node_names.data(),
                                      outputSize);
}



