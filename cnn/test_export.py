import numpy as np
import onnx_chainer
import chainercv.links as C
import chainer.links as L
import chainer
from chainer import training, iterators, serializers, optimizers
import Comparator1
import CNNf_net
import paint
import os

Resume="D:/C_code/1personal presentation/result/snapshot_iter_31250"

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

train, test = chainer.datasets.get_cifar100()

#model = C.ResNet101(n_class=100)
#model = C.ResNet101(pretrained_model='imagenet')
model=CNNf_net.CNNM(100)
model.to_gpu()

#serializers.load_npz(Resume, model)
serializers.load_npz("D:/C_code/1personal presentation/store/cnn-cifar100_pro.model", model)

#model.to_cpu()


print(chainer.cuda.available)

with chainer.using_config('train', False):
    Comparator1.predict_cifar(model, test)

# x = chainer.Variable(np.zeros((1, 3, 32, 32), dtype=np.float32))
# x.to_gpu()
# onnx_chainer.export(model, x, filename='cnn-cifar100_pro_t.onnx')

# basedir = "D:/C_code/1personal presentation/images"
# paint.plot_predict_cifar(os.path.join(basedir, 'cifar100_predict.png'), model,
#                        test, 4, 5, scale=5., label_list=CIFAR100_LABELS_LIST)