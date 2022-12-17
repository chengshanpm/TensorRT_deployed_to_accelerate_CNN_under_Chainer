from chainer import Variable, cuda
import time
import numpy as np
import chainer
import Inference
import CNNf_net

def predict_cifar(model, data):
    time_start = time.time()
    h = 0
    for i in range(len(data)):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        xp = cuda.cupy
        x = Variable(xp.asarray(image.reshape(1, 3, 32, 32)))    # test data
        #x = Variable(np.asarray(image.reshape(1, 3, 32, 32)))
        #t = Variable(xp.asarray([test[i][1]])) # labels
        y = model(x)                              # Inference result
        #根据置信度选择最大值即为对应的class
        prediction = y.data.argmax(axis=1)
        # print('Predicted {}-th image, prediction={}, actual={}'
        #       .format(i, prediction[0], label_index))
        if prediction[0]==label_index :
            h = h + 1
    acc = h/len(data)
    time_end = time.time()
    time_cost = time_end - time_start
    print()
    print('accuracy={}, time_cost={}'
              .format(acc, time_cost))
    return acc, time_cost

def predict_cifar_tensorrt(engine, data):
    time_start = time.time()
    h = 0
    for i in range(len(data)):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        xp = cuda.cupy
        #x = xp.asarray(image.reshape(1, 3, 32, 32))    # test data
        x = np.asarray(image.reshape(1, 3, 32, 32))
        #t = Variable(xp.asarray([test[i][1]])) # labels
        y = Inference.infer(engine, x)      # Inference result
        #根据置信度选择最大值即为对应的class
        prediction = np.argmax(y)
        # print('Predicted {}-th image, prediction={}, actual={}'
        #       .format(i, prediction[0], label_index))
        if prediction==label_index :
            h = h + 1
    acc = h/len(data)
    time_end = time.time()
    time_cost = time_end - time_start
    print()
    print('accuracy={}, time_cost={}'
              .format(acc, time_cost))
    return acc, time_cost

if __name__ == "__main__":
    _, test = chainer.datasets.get_cifar100()
    model=CNNf_net.CNNM(100)
    model.to_gpu()
    chainer.serializers.load_npz("D:/C_code/1personal presentation/store/cnn-cifar100_pro.model", model)
    with chainer.using_config('train', False):
        acc1, time_cost1 = predict_cifar(model, test)
    print()
    engine = Inference.load_engine("cnn_cifar100.engine")
    acc2, time_cost2 = predict_cifar_tensorrt(engine, test)
    print()
    print('accuracy_cnn={}, accuracy_cnn_tensor={}'
              .format(acc1, acc2))
    print('time_cost_cnn={}, time_cost_cnn_tensor={}'
              .format(time_cost1, time_cost2))


