import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

np.seterr(over='ignore')

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = np.dot(sigmoid(z).T, (1 - sigmoid(z)))
    return g


def yvec(y, out_layer_size):
    m = y.shape[0]
    yvec = np.zeros((m, out_layer_size))
    for k in range(m):
        yvec[k, int(y[k])] = 1
    return yvec


def paramsToTheta(nn_params, input_layer_size, hidden_layer_size, out_layer_size):
    Theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))

    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
        out_layer_size, hidden_layer_size + 1)

    return Theta1, Theta2


def forwardProp(Theta1, Theta2, X):
    
    m, n = X.shape    

    a1 = np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack((np.ones((m,1)), sigmoid(z2)))
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)  

    return a3, z2
    


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   out_layer_size,
                   X, y, lmbd):

    m, n = X.shape 

    Theta1, Theta2 = paramsToTheta(nn_params, input_layer_size, hidden_layer_size, out_layer_size)

    a3 = forwardProp(Theta1, Theta2, X)

    yv = yvec(y, out_layer_size)

    J = 0

    for i in range(out_layer_size):
        cost0 = np.dot((1 - yv[:, i]).T, np.log(1 - a3[:, i]))
        cost1 = np.dot(yv[:, i].T, np.log(a3[:, i]))
        J = J - 1 * (cost0 + cost1) / m

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
    return J

def nnGrad(nn_params,
           input_layer_size,
           hidden_layer_size,
           out_layer_size,
           X, y, lmbd):

    # BACK PROPAGATION

    m, n = X.shape

    X1 = np.hstack((np.ones((m,1)),X))

    Theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(
        hidden_layer_size, (input_layer_size + 1))

    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
        out_layer_size, hidden_layer_size + 1)

    #regularization
    Theta1reg = Theta1
    Theta1reg[:, 0] = np.zeros((1, hidden_layer_size))
    Theta2reg = Theta2
    Theta2reg[:, 0] = np.zeros((1, out_layer_size))

    #y vectorization
    yv = yvec(y, out_layer_size)

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

    return grad

def learn():
    Theta1 = np.random.random((input_layer_size+1, hidden_layer_size)) * 2 * 0.12 - 0.12
    Theta2 = np.random.random((hidden_layer_size+1, out_layer_size)) * 2 * 0.12 - 0.12
    nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))

    print nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   out_layer_size,
                   X, y, lmbd)

    res1 = np.array(nn_params.shape)
    args = (input_layer_size, hidden_layer_size, out_layer_size, X, y, lmbd)
    res1 = optimize.fmin_cg(nnCostFunction, nn_params, 
                            fprime=nnGrad, args=args, retall=0,
                            maxiter=300)
    return res1


def check(nn_params):
    Theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(
            hidden_layer_size, (input_layer_size + 1))

    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
            out_layer_size, hidden_layer_size + 1)

    np.savetxt('digits\\T1_out.csv', Theta1, delimiter=',')
    np.savetxt('digits\\T2_out.csv', Theta2, delimiter=',')

    a3 = forwardProp(Theta1, Theta2, X)
    y_res = np.argmax(a3, axis=1).reshape((m,1))
    print float((y==y_res).sum())/m


"""X = np.loadtxt('digits\\X_train.csv', delimiter=',')
y = np.loadtxt('digits\\y_train.csv', delimiter=',')
m, n = X.shape
input_layer_size = n
hidden_layer_size = 25
out_layer_size = 10
lmbd = 1

nn_params = learn()
check(nn_params)"""