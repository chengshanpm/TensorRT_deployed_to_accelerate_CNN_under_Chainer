import chainer
import chainer.functions as F
import chainer.links as L


class CNNM(chainer.Chain):
    def __init__(self, n_out):
        super(CNNM, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3)
            self.conv2 = L.Convolution2D(64, 64, 3)
            self.conv3 = L.Convolution2D(64, 128, 3)
            self.conv4 = L.Convolution2D(128, 128, 3)
            self.conv5 = L.Convolution2D(128, 256, 3)
            self.conv6 = L.Convolution2D(256, 256, 3)
            self.conv7 = L.Convolution2D(256, 512, 3)
            self.conv8 = L.Convolution2D(512, 512, 3)
            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(64)
            self.bn3 = L.BatchNormalization(128)
            self.bn4 = L.BatchNormalization(128)
            self.bn5 = L.BatchNormalization(256)
            self.bn6 = L.BatchNormalization(256)
            # self.bn7 = L.BatchNormalization(512)
            # self.bn8 = L.BatchNormalization(512)
            self.pool1 = _max_pooling_2d
            self.pool2 = _max_pooling_2d
            self.pool3 = _max_pooling_2d
            self.pool4 = _max_pooling_2d
            self.pool5 = _max_pooling_2d
            self.fc1 = L.Linear(None, 256)
            self.fc2 = L.Linear(256, 128)
            self.fc3 = L.Linear(128, n_out)
 
    def __call__(self, x):
        #h = F.elu(self.conv1(x))
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        #h = F.relu(self.conv1(x))
        #h = self.pool1(h)
        #h = F.relu(self.conv2(h))
        h = self.pool2(h)
        #h = F.max_pooling_2d(h, 2, 2)
        #h = F.relu(self.conv3(h))
        #h = F.relu(self.conv4(h))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        # h = F.max_pooling_2d(h, 2, 2)
        h = self.pool3(h)
        #h = F.relu(self.conv5(h))
        #h = F.relu(self.conv6(h))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))
        h = self.pool4(h)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)
        # h = F.relu(self.bn7(self.conv7(h)))
        # h = F.relu(self.bn8(self.conv8(h)))
        #h = self.pool5(h)
        # h = F.max_pooling_2d(h, 2, 2)
        # h = F.dropout(h, ratio=0.25)
        h = F.relu(self.fc1(h))
        # h = F.dropout(h, ratio=0.1)
        # h = F.dropout(h)
        h = F.relu(self.fc2(h))
        # h = F.dropout(h, ratio=0.1)
        # h = F.dropout(h)
        h = self.fc3(h)
        return h
    
def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)
    # F.max_pooling_2d(x, ksize=2， stride=2)