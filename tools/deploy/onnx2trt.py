# coding=utf-8  
# @Time   : 2020/11/23 10:12
# @Auto   : zzf-jeff

import sys

sys.path.append('../')
import argparse
import tensorrt as trt

from .utils import gen_ones

def _onnx2trt(model, log_level='ERROR', max_batch_size=1, max_workspace_size=1,
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
    if isinstance(model, str):
        with open(model, 'rb') as f:
            flag = parser.parse(f.read())
    else:
        flag = parser.parse(model.read())
    if not flag:
        for error in range(parser.num_errors):
            print(parser.get_error(error))

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
    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        if int8_calibrator is None:
            shapes = [(1,) + network.get_input(i).shape[1:]
                      for i in range(network.num_inputs)]
            dummy_data = gen_ones(shapes)
            int8_calibrator = EntropyCalibrator2(CustomDataset(dummy_data))
        config.int8_calibrator = int8_calibrator



def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifiers model')
    parser.add_argument('--model', type=str, help='train config file path', default='config/resnet50.yaml')
    parser.add_argument('--output', type=str, help='onnx save path')
    args = parser.parse_args()
    return args


def main():
    pass


if __name__ == '__main__':
    main()
