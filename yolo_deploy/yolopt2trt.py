import tensorrt as trt


def onnx2trt(onnx_model_path, tensorrt_model_path, mode):
    assert mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \
                                             "but got {}".format(mode)
    trt_logger = trt.Logger(getattr(trt.Logger, 'INFO'))
    builder = trt.Builder(trt_logger)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)  # 创建一个显式批次的网络
    parser = trt.OnnxParser(network, trt_logger)  # 将网络和解析器绑定在一起，这样解析器就知道要解析哪个网络了
    with open(onnx_model_path, 'rb') as f:
        flag = parser.parse(f.read())
    if not flag:
        for error in range(parser.num_errors):
            print(parser.get_error(error))
    # output_tensors = [network.get_output(i) for i in range(network.num_outputs)]  # 获取网络中所有的输出张量
    # # [network.unmark_output(tensor) for tensor in output_tensors]
    # for tensor in output_tensors:
    #     identity_out_tensor = network.add_identity(tensor).get_output(0)
    #     identity_out_tensor.name = 'identity_{}'.format(tensor.name)
    #     network.mark_output(tensor=identity_out_tensor)  # 将identity_out_tensor标记为网络的输出张量
    config: trt.IBuilderConfig = builder.create_builder_config()
    # config.set_memory_pool_limit(1 << 25)
    if mode == 'fp16':
        assert builder.platform_has_fast_fp16, "not support fp16"
        config.set_flag(trt.BuilderFlag.FP16)
    if mode == 'int8':
        assert builder.platform_has_fast_int8, "not support int8"
        # config.set_flag(trt.BuilderFlag.INT8)
        # config.int8_calibrator = trt_int8_calibrator
        # print(trt_int8_calibrator)
    engine = builder.build_serialized_network(network, config)
    with open(tensorrt_model_path, 'wb') as f:
        f.write(engine)

if __name__ == "__main__":
    onnx2trt("../onnx_model/yolov8x.onnx","../trt_model/yolov8x_32.trt","fp32")
