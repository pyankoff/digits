import csv
import numpy as np
import matplotlib.pyplot as plt
from digits import nnCostFunction, sigmoid, yvec, sigmoidGradient
from math import log



input_layer_size = 400
hidden_layer_size = 25
out_layer_size = 10

Theta1 = np.loadtxt('digits\\testing\\1_test.csv', delimiter=',')
Theta2 = np.loadtxt('digits\\testing\\2_test.csv', delimiter=',')
X = np.loadtxt('digits\\testing\\X_train.csv', delimiter=',')
y = np.loadtxt('digits\\testing\\y_train.csv', delimiter=',')
lmbd = 0

m, n = X.shape    

#a1 
a1 = np.hstack((np.ones((m,1)),X))
z2 = np.dot(a1, Theta1.T)
a2 = np.hstack((np.ones((m,1)), sigmoid(z2)))
z3 = np.dot(a2, Theta2.T)
a3 = sigmoid(z3)   
print a3
yv = yvec(y, out_layer_size)

J = 0
for i in range(m):
    for k in range(out_layer_size):
        cost0 = -yv[i,k] * log(a3[i,k])
        cost1 = -(1 - yv[i,k]) * log(1 - a3[i,k])
        J = J + (cost0 + cost1)/m

#regularization
Theta1reg = Theta1
Theta1reg[:, 0] = np.zeros((1, hidden_layer_size))
Theta2reg = Theta2
Theta2reg[:, 0] = np.zeros((1, out_layer_size))

t1reg = (Theta1reg ** 2).sum()
t2reg = (Theta2reg ** 2).sum()

reg = 0.5 * lmbd * (t1reg + t2reg) / m

J = J + reg

print J



"""# BACK PROPAGATION


X1 = np.hstack((np.ones((m,1)),X))

D1 = np.zeros(Theta1.shape)
D2 = np.zeros(Theta2.shape)

for t in xrange(m):
    # step 1
    a1 = X1[t, :].reshape(n+1, 1) 
    z2 = np.dot(Theta1, a1).reshape(hidden_layer_size,1)
    a2 = np.vstack((np.ones((1,1)), sigmoid(z2)))
    z3 = np.dot(Theta2, a2).reshape(out_layer_size,1)
    a3 = sigmoid(z3)   

    # step 2
    yt = yv[t, :].reshape(out_layer_size,1)
    delta3 = a3 - yt;         

    # step 3
    delta2 = np.dot(Theta2[:, 1:].T, delta3) * sigmoidGradient(z2)


    # step 4
    D1 = D1 + np.dot(delta2, a1.T)
    D2 = D2 + np.dot(delta3, a2.T)

Theta1_grad = D1 / m + lmbd * Theta1reg / m
Theta2_grad = D2 / m + lmbd * Theta2reg / m 

grad = np.hstack((Theta1_grad.ravel(), Theta2_grad.ravel()))

print grad"""