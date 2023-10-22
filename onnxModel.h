//
// Created by guo on 23-9-25.
//

#ifndef DEPLOY_ONNXMODEL_H
#define DEPLOY_ONNXMODEL_H

#include "onnxruntime_cxx_api.h"
class onnxModel {
public:
    Ort::Session session = Ort::Session(nullptr);
    size_t num_input_nodes;
    size_t num_output_nodes;
    std::vector<const char *> input_node_names;
    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> input_nodes_dims;
    std::vector<std::vector<int64_t>> output_nodes_dims;

    onnxModel(Ort::Env &env, Ort::SessionOptions &session_options, const char *model_path);
    void putInputNames(std::vector<const char *> inputNames);
    void putOutputNames(std::vector<const char *> outputNames);
    Ort::Value run(Ort::Value & input_tensor,size_t inputSize,size_t outputSize);

};


#endif //DEPLOY_ONNXMODEL_H