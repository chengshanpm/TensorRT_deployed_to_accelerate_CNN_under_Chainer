# 导入必用依赖
import tensorrt as trt
import pycuda.autoinit  #负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  #GPU CPU之间的数据传输
import numpy as np
import os
import chainer

engine_file = "cnn_cifar100.engine"

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 创建logger：日志记录器
# logger = trt.Logger(trt.Logger.WARNING)
#logger = trt.Logger(trt.Logger.INFO)
TRT_LOGGER = trt.Logger()

# 创建runtime并反序列化生成engine
# engine = load_engine(engine_file)
# with open("cnn_cifar100.engine", "rb") as f, trt.Runtime(logger) as runtime:
#     engine=runtime.deserialize_cuda_engine(f.read())


def infer(engine, input_image):
    image_width = 32
    image_height = 32

    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
        # Allocate host and device buffers 分配CPU锁页内存和GPU显存
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        # 创建cuda流
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
        return output_buffer

if __name__ == "__main__":
    print("Running TensorRT inference for CNN")
    _, test = chainer.datasets.get_cifar100()
    image, _ = test[0]
    image = np.array(image)
    image_o = image.reshape(1,3,32,32)

    with load_engine(engine_file) as engine:
        print(infer(engine, image_o))








# h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
# h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
# d_input = cuda.mem_alloc(h_input.nbytes)
# d_output = cuda.mem_alloc(h_output.nbytes)


# 创建context并进行推理
# with engine.create_execution_context() as context:
#     # Transfer input data to the GPU.
#     cuda.memcpy_htod_async(d_input, h_input, stream)
#     # Run inference.
#     context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
#     # Transfer predictions back from the GPU.
#     cuda.memcpy_dtoh_async(h_output, d_output, stream)
#     # Synchronize the stream
#     stream.synchronize()
#     # Return the host output. 该数据等同于原始模型的输出数据
#     return h_output


