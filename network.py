# the original model
import matplotlib.pyplot as plt
import numpy as np

class network(object):
    def __init__(self, f_num,col_num):

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

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        W1 += -epsilon * dX[0]
        b1 += -epsilon * dX[1]
        W2 += -epsilon * dX[2]
        b2 += -epsilon * dX[3]
        self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'] = W1, b1, W2, b2