# 60000条训练数据和10000条测试数据，28x28像素的灰度图像
# 输入层有784个节点，隐含层有40个神经元，输出层有10个节点
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class network(object):
    def __init__(self, f_num,col_num):

        # input X:example_num*f_num; col_num different class

        self.f_num = f_num
        self.col_num = col_num

        W1 = np.random.randn(f_num, 128)
        b1 = np.random.randn(1, 128)
        W2 = np.random.randn(128, col_num)
        b2 = np.random.randn(1, col_num)

        self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


    def forward(self, X, y,y_origin):
        self.y = y
        self.y_origin=y_origin
        self.example_num=X.shape[0]
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        self.z1 = X.dot(W1) + b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1.dot(W2) + b2
        self.exp_scores = np.exp(self.z2)
        self.probs = self.exp_scores / np.sum(self.exp_scores, axis=1, keepdims=True)
        loss = -(self.y * np.log(self.probs+0.0000000001)).sum() / self.example_num
        return loss

    def backward(self, X):

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        delta3 = self.probs
        delta3[range(self.example_num), self.y_origin] -= 1
        dW2 = (self.a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        return dW1, db1, dW2, db2

    def update_net(self, dX, epsilon):

        # dX:[dW1,db1,dW2,db2]

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        W1 += -epsilon * dX[0]
        b1 += -epsilon * dX[1]
        W2 += -epsilon * dX[2]
        b2 += -epsilon * dX[3]
        self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'] = W1, b1, W2, b2

# 加载数据
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

X_train, X_test = tf.cast(train_x / 255.0, tf.float32), tf.cast(test_x / 255.0, tf.float32)  # 归一化
X_train_flat = tf.reshape(X_train, (-1, 28*28))
X_train_flat=X_train_flat.numpy()
X_test_flat = tf.reshape(X_test, (-1, 28*28))
X_test_flat=X_test_flat.numpy()
y_train, y_test = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s' % (train_x.shape, train_y.shape, test_x.shape, test_y.shape))
y_train_onehot = tf.keras.utils.to_categorical(train_y, 10)
y_test_onehot = tf.keras.utils.to_categorical(test_y, 10)
print('\n X_train_flat:%s, y_train_onehot:%s, X_test_flat:%s, y_test_onehot:%s' %
      (X_train_flat.shape, y_train_onehot.shape, X_test_flat.shape, y_test_onehot.shape))
print(y_train_onehot[2])
def main():
    net=network(28*28,10)
    for epoch in range(500):#100时accuracy为33%左右  500时accuracy为62%左右 将神经元数量从40改为128后accuracy为72%，但速度大大下降；
        loss=net.forward(X_train_flat,y_train_onehot,y_train)
        dW1, db1, dW2, db2 = net.backward(X_train_flat)
        net.update_net([dW1, db1, dW2, db2],epsilon=0.00000001)
        print(f'Epoch{epoch},loss:{loss}')

    net.forward(X_test_flat,y_test_onehot,y_train)
    predictions=np.argmax(net.probs,axis=1)
    accuracy=np.mean(predictions==y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()




