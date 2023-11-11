//
// Created by fwq on 23-11-9.
//
#include<iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
class Logger : public ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        std::cout << msg << std::endl;
    }
} logger;
void yoloonnx2trt(int batch_size){
    IBuilder* builder = createInferBuilder(logger);
    IBuilderConfig* config = builder->createBuilderConfig();

    INetworkDefinition* network = builder->createNetworkV2(1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IParser* onnx_parser = createParser(*network,logger);


}