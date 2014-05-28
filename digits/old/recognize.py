import csv
import numpy as np
import matplotlib.pyplot as plt
from digits import nnCostFunction



input_layer_size = 400
hidden_layer_size = 25
out_layer_size = 10

Theta1 = np.loadtxt('digits\\1_test.csv', delimiter=',')
Theta2 = np.loadtxt('digits\\2_test.csv', delimiter=',')
X = np.loadtxt('digits\\X_train.csv', delimiter=',')
y = np.loadtxt('digits\\y_train.csv', delimiter=',')
nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))
lmbd = 0

print nnCostFunction(nn_params,
                    input_layer_size,
                    hidden_layer_size,
                    out_layer_size,
                    X, y, lmbd)






#show_num(X, y, point)

def show_num(X, y, i):
    show(X[i].reshape(28, 28))


def show(X):
    plt.imshow(X, cmap=plt.cm.gray_r)
    plt.show()