#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <iterator>
#include "NvOnnxParser.h"


using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        std::cout << msg << std::endl;
    }
} logger;


int main() {
    std::ifstream file("/opt/deploy/trt_model/a1.trt", std::ios::binary | std::ios::ate);
    const int BATCH_SIZE = 1;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }
    auto runtime = std::unique_ptr<IRuntime>{createInferRuntime(logger)};
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    size_t ioTensorsCount = engine->getNbIOTensors();
    std::vector<void*> d_tensor(ioTensorsCount);

    std::vector<Dims3> inputShapes;
    std::vector<Dims3> outputShapes;
    cudaStream_t stream;
    for(int i = 0; i < ioTensorsCount; i++){
        auto name = engine->getIOTensorName(i);
        auto tensorShape = engine->getTensorShape(name);
        auto tensorMode = engine->getTensorIOMode(name);
        cudaMallocAsync(&d_tensor[i],BATCH_SIZE*tensorShape.d[1]*tensorShape.d[2]*tensorShape.d[3]* sizeof(float),stream);

        if (tensorMode == TensorIOMode::kINPUT)
            inputShapes.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
        else if (tensorMode == TensorIOMode::kOUTPUT)
            outputShapes.emplace_back(tensorShape.d[1],tensorShape.d[2],tensorShape.d[3]);
    }


    return 0;

}