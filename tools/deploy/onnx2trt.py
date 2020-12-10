# coding=utf-8  
# @Time   : 2020/11/23 10:12
# @Auto   : zzf-jeff

import sys

sys.path.append('./')
import argparse
import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def build_engine(onnx_file_path, engine_file_path, batch_size=4, mode='fp32'):
    ## step1:创建trt的logger对象
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    ## 如果存在engine，则直接返回
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        ## 从序列化engine文件中创建推理引擎
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    ## trt 7+ 需要添加的
    ## step2: 创建trt的builder对象,
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_batch_size = int(batch_size)
        builder.fp16_mode = True if mode == 'fp16' else False
        builder.int8_mode = True if mode == 'int8' else False
        builder.strict_type_constraints = True
        builder.max_workspace_size = 1 << 32  # 1GB:30

        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        # for i in range(network.num_inputs):
        #     tensor = network.get_input(i)
        #     name = tensor.name
        #     shape = tensor.shape[1:]
        #     min_shape = (1,) + shape
        #     opt_shape = ((1 + batch_size) // 2,) + shape
        #     max_shape = (batch_size,) + shape
        #     profile.set_shape(name, min_shape, opt_shape, max_shape)
        #
        profile.set_shape(network.get_input(0).name, (batch_size, 3, 224, 224))

        config.add_optimization_profile(profile)

        print('network layers is', len(network))  # Printed output == 0. Something is wrong.
        last_layer = network.get_layer(network.num_layers - 1)
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
        engine = builder.build_cuda_engine(network, config)

        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
    return engine


def _onnx2trt(model, engine_output, log_level='ERROR', max_batch_size=1, max_workspace_size=1,
              fp16_mode=False,
              strict_type_constraints=False,
              int8_mode=False,
              int8_calibrator=None):
    """build TensorRT model from Onnx model.

    Args:
        model (string or io object): Onnx model name
        log_level (string, default is ERROR): TensorRT logger level, now
            INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        max_batch_size (int, default=1): The maximum batch size which can be
            used at execution time, and also the batch size for which the
            ICudaEngine will be optimized.
        max_workspace_size (int, default is 1): The maximum GPU temporary
            memory which the ICudaEngine can use at execution time. default is
            1GB.
        fp16_mode (bool, default is False): Whether or not 16-bit kernels are
            permitted. During engine build fp16 kernels will also be tried when
            this mode is enabled.
        strict_type_constraints (bool, default is False): When strict type
            constraints is set, TensorRT will choose the type constraints that
            conforms to type constraints. If the flag is not enabled higher
            precision implementation may be chosen if it results in higher
            performance.
        int8_mode (bool, default is False): Whether Int8 mode is used.
        int8_calibrator (volksdep.calibrators.base.BaseCalibrator,
            default is None): calibrator for int8 mode, if None, default
            calibrator will be used as calibration data.
    """

    logger = trt.Logger(getattr(trt.Logger, log_level))
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(model, 'rb') as f:
        # flag = parser.parse(f.read())
        parser.parse(f.read())
    # if not flag::
    #         print(parser.get_error(error))
    #     for error in range(parser.num_errors)

    # re-order output tensor
    output_tensors = [network.get_output(i)
                      for i in range(network.num_outputs)]
    [network.unmark_output(tensor) for tensor in output_tensors]
    for tensor in output_tensors:
        identity_out_tensor = network.add_identity(tensor).get_output(0)
        identity_out_tensor.name = 'identity_{}'.format(tensor.name)
        network.mark_output(tensor=identity_out_tensor)

    builder.max_batch_size = max_batch_size
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size * (1 << 25)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape[1:]
        min_shape = (1,) + shape
        opt_shape = ((1 + max_batch_size) // 2,) + shape
        max_shape = (max_batch_size,) + shape
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)

    # trt_model = TRTModel(engine)
    with open(engine_output, 'wb') as f:
        f.write(engine.serialize())

    # return trt_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--onnx_input', type=str, help='train config file path')
    parser.add_argument('--engine_output', type=str, help='onnx save path')
    parser.add_argument('--source', type=str, help='test img path')
    args = parser.parse_args()
    return args


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main():
    args = parse_args()
    onnx_input = args.onnx_input
    engine_output = args.engine_output
    img_path = args.source
    # _onnx2trt(onnx_input,engine_output)
    engine = build_engine(onnx_file_path=onnx_input, engine_file_path=engine_output)
    # inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    img = Image.open(args.source)
    img = img.convert('RGB')
    img = trans(img)
    img = img.unsqueeze(0)
    print(img.shape)

    h, w = img.shape[2:4]

    img_numpy = np.array(img, dtype=np.float32, order='C')

    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    cuda.memcpy_htod_async(device_input, img_numpy, stream)
    start_time = time.time()
    print(engine.max_batch_size)
    for _ in range(1):
        # run inference
        context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()
        # postprocess results

        output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])
    end_time = time.time()

    print(end_time - start_time)
    prob = F.softmax(output_data, dim=1)
    value, predicted = torch.max(output_data.data, 1)
    pred_class = ['dog', 'cat'][predicted.item()]
    pred_score = prob[0][predicted.item()].item()
    print(pred_class, pred_score)


if __name__ == '__main__':
    main()
