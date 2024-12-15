import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
import numpy as np

class network(object):
    def __init__(self, f_num,col_num):
        self.f_num = f_num
        self.col_num = col_num
        W1 = np.random.randn(f_num, 40)
        b1 = np.random.randn(1, 40)
        W2 = np.random.randn(40, col_num)
        b2 = np.random.randn(1, col_num)
        self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def forward(self, X, y):
        self.y = y
        self.example_num = X.shape[0]
        self.y_ext = np.zeros((self.example_num, self.col_num))
        for i, j in enumerate(y, 0):
            self.y_ext[i, j] = 1

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        self.z1 = X.dot(W1) + b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1.dot(W2) + b2
        self.exp_scores = np.exp(self.z2)
        self.probs = self.exp_scores / np.sum(self.exp_scores, axis=1, keepdims=True)
        loss = -(self.y_ext * np.log(self.probs)).sum() / self.example_num
        return loss

    def backward(self, X):

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        delta3 = self.probs
        delta3[range(self.example_num), self.y] -= 1
        dW2 = (self.a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        return dW1, db1, dW2, db2

    def update_net(self, dX, epsilon=0.001):

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        W1 += -epsilon * dX[0]
        b1 += -epsilon * dX[1]
        W2 += -epsilon * dX[2]
        b2 += -epsilon * dX[3]
        self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'] = W1, b1, W2, b2

X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def predict(model,x):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    z1=x.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    exp_scores=np.exp(z2)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    return np.argmax(probs,axis=1)

def get_acc(model,x,y):
    y_pred=predict(model,x)
    return (y_pred==y).sum()/x.shape[0]


if __name__=='__main__':
    net=network(2,2)
    i=0
    best_acc=0
    best_model={}
    while True:
        loss=net.forward(X,y)
        acc=get_acc(net.model,X,y)
        if best_acc<acc:
            best_acc=acc
            best_model=net.model
        print('%d loss:'%(i+1),loss,' acc:',acc)
        if acc>0.99:
            break
        if i>100000:
            break
        dW1,db1,dW2,db2=net.backward(X)
        net.update_net([dW1,db1,dW2,db2])
        i+=1

    print('%d iteration,best_acc: %f'%(i+1,best_acc),'\n','model:',best_model)
    plot_decision_boundary(lambda x: predict(best_model,x))
    plt.show()

    ###