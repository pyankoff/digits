import numpy as np
from learning_class import neuralNetwork

np.seterr(over='ignore')

def checkCost():
    Theta1 = np.loadtxt('digits\\testing\\1_test.csv', delimiter=',')
    Theta2 = np.loadtxt('digits\\testing\\2_test.csv', delimiter=',')
    nn_params = np.hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

    hidden_layer_size = 25
    out_layer_size = 10
    X = np.loadtxt('digits\\testing\\X_train.csv', delimiter=',')
    y = np.loadtxt('digits\\testing\\y_train.csv', delimiter=',')
    lmbd = 1

    a = neuralNetwork(nn_params, hidden_layer_size, out_layer_size,
                      X, y, lmbd)

    print "Cost: ", a.nnCostFunction(nn_params)   



def checkGrad():
    
    def debugInit(fan_out, fan_in):
        W = np.zeros((fan_out, 1 + fan_in))
        W = np.sin(1 + np.arange(W.size)) / 10
        return W

    input_layer_size = 3
    hidden_layer_size = 5
    out_layer_size= 3
    m = 5
    lmbd = 0

    # We generate some 'random' test data
    Theta1 = debugInit(hidden_layer_size, input_layer_size).reshape(
        hidden_layer_size, 1+input_layer_size)
    Theta2 = debugInit(out_layer_size, hidden_layer_size).reshape(
        out_layer_size, 1+hidden_layer_size)
    nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))

    # Reusing debugInitializeWeights to generate X
    X  = debugInit(m, input_layer_size - 1).reshape(m, input_layer_size, order='F')
    y  = np.array([[1,2,0,1,2]]).T

    a = neuralNetwork(nn_params, hidden_layer_size, out_layer_size,
                      X, y, lmbd)

    #print "Theta1: \n", Theta1
    #print "Theta2 \n", Theta2
    #print "X: \n", X
    #print "nn_params: \n", nn_params
    print "Grad: \n", a.nnGrad(nn_params)
    print "Numerical grad: \n", a.computeNumGradient()


def checkLearn():
    Theta1 = np.loadtxt('digits\\testing\\T1_out.csv', delimiter=',')
    Theta2 = np.loadtxt('digits\\testing\\T2_out.csv', delimiter=',')
    nn_params = np.hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

    hidden_layer_size = 25
    out_layer_size = 10
    X = np.loadtxt('digits\\testing\\X_train.csv', delimiter=',')
    y = np.loadtxt('digits\\testing\\y_train.csv', delimiter=',')
    lmbd = 1

    a = neuralNetwork(nn_params, hidden_layer_size, out_layer_size,
                      X, y, lmbd)

    #a.learn()
    a.check()

def learnKaggleData():
    X = np.loadtxt('digits\\X_full.csv', delimiter=',')
    y = np.loadtxt('digits\\y_full.csv', delimiter=',')
    #y = np.zeros(28000)

    hidden_layer_size = 25
    out_layer_size = 10
    lmbd = 0.1

    nn_params = np.zeros(785*hidden_layer_size +
                         (hidden_layer_size+1)*out_layer_size)
    Theta1 = np.loadtxt('digits\\T1_out.csv', delimiter=',')
    Theta2 = np.loadtxt('digits\\T2_out.csv', delimiter=',')
    nn_params = np.hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

    a = neuralNetwork(nn_params, hidden_layer_size, out_layer_size,
                      X, y, lmbd)

    #a.learn()
    a.look()


learnKaggleData()
