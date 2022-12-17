import onnx
import numpy as np
import onnxruntime as rt
import chainer
import cv2
from chainer import serializers
import CNNf_net

model_path = 'D:/C_code/1personal presentation/store/cnn-cifar100_pro_t.onnx'
chainer_mode=0

# 验证模型合法性
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
# if chainer_mode==1:
#     model=CNNf_net.CNNM(100)
#     serializers.load_npz("D:/C_code/1personal presentation/store/cnn-cifar100_pro.model", model)
#     model.to_gpu()

#onnx_model.ir_version=6
# 读入图像并调整为输入维度
# image = cv2.imread("data/images/person.png")
# image = cv2.resize(image, (448,448))
# image = image.transpose(2,0,1)
# image = np.array(image)[np.newaxis, :, :, :].astype(np.float32)
_, test = chainer.datasets.get_cifar100()
image, _ = test[0]
image = np.array(image)
image_o = image.reshape(1,3,32,32)
# if chainer_mode==1:
#     image_c=chainer.Variable(image.reshape(1,3,32,32))
#     image_c.to_gpu()
#     with chainer.using_config('train', False):
#         print(model(image_c))

# 设置模型session以及输入信息
sess = rt.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
#input_name2 = sess.get_inputs()[1].name
#input_name3 = sess.get_inputs()[2].name

print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output = sess.run(None, {input_name: image_o})
print(output)


