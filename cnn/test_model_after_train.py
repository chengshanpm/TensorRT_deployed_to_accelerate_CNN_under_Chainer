import Comparator1
import CNNf_net
from chainer import  serializers
import chainer


train, test = chainer.datasets.get_cifar100()

model=CNNf_net.CNNM(100)
model.to_gpu()
serializers.load_npz("D:/C_code/1personal presentation/result_show/cnnm-cifar100.model", model)

with chainer.using_config('train', False):
    Comparator1.predict_cifar(model, test)

