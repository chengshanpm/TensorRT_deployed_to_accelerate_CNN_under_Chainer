#import numpy as np

import chainer
#import chainercv.links as C
import chainer.links as L
from chainer import training, iterators, serializers, optimizers
from chainer.training import extensions
import onnx_chainer
#import Comparator1
import CNNf_net
#import CNNm_net

Batch_Size=32
Epoch=80
Out="D:/C_code/1personal presentation/result_show"
#Resume="D:/C_code/1personal presentation/result/snapshot_iter_6250"

# model = C.VGG16(n_class=100)
# model = C.ResNet50(n_class=100)
model=CNNf_net.CNNM(100)
# model = CNNm_net.CNNMedium(100)
#serializers.load_npz("D:/C_code/1personal presentation/store/no_cnn-cifar100_pro.model", model)
classifier_model = L.Classifier(model)
gpu=0

if gpu >= 0:
    chainer.cuda.get_device(gpu).use()  # Make a specified GPU current
    classifier_model.to_gpu()  # Copy the model to the GPU

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

optimizer = optimizers.Adam()
optimizer.setup(classifier_model)

train, test = chainer.datasets.get_cifar100()
#Comparator1.predict_cifar(model, test, label_list=CIFAR100_LABELS_LIST)

train_iter = iterators.SerialIterator(train, Batch_Size)
# test_iter = iterators.SerialIterator(test, Batch_Size,
#                                      repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

trainer = training.Trainer(updater, (Epoch, 'epoch'), out=Out)

# 该方法是用来作为 validation验证集在训练的时候调超参数用的（但目前这个没有调整超参数这个考量，所以就不需要验证集）
# 所以validation 不需要的话，print那里也可以去掉这个（因为一开始是用了所以有一组数据，但是从epoch=2开始就没有了）
# trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=gpu))

trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    x_key='epoch',
    file_name='accuracy.png'))

trainer.extend(extensions.ProgressBar())

# if Resume:
#     serializers.load_npz(Resume, trainer)

trainer.run()

serializers.save_npz('{}/{}-cifar100.model'
                         .format(Out, 'cnnm'), model)


# # Pseudo input
# x = chainer.Variable(np.zeros((1, 3, 32, 32), dtype=np.float32))
# x.to_gpu()
# # export
# onnx_chainer.export(model, x, filename='cnnm_cifar100.onnx')




