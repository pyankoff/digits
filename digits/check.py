import numpy as np
from learning_class_2hidden import neuralNetwork

np.seterr(over='ignore')


def nn_train():    
    input_layer_size = 784
    hidden_layer1_size = 250
    hidden_layer2_size = 250
    out_layer_size = 10
    lmbd = 0.3


    Theta1 = np.loadtxt('T1_out.csv', delimiter=',')
    Theta2 = np.loadtxt('T2_out.csv', delimiter=',')
    Theta3 = np.loadtxt('T3_out.csv', delimiter=',')
    nn_params = np.hstack((Theta1.ravel(order='F'), 
                          Theta2.ravel(order='F'),
                         Theta3.ravel(order='F')))

    """params_size = (input_layer_size + 1) * hidden_layer1_size \
              + (hidden_layer1_size + 1) * hidden_layer2_size \
              + (hidden_layer2_size + 1) * out_layer_size    
    nn_params = np.random.random(params_size) * 2 * 0.12 - 0.12"""

    X = np.loadtxt('X_5000_3.csv', delimiter=',')
    
    y = np.loadtxt('y_5000_3.csv', delimiter=',')

    a = neuralNetwork(nn_params, hidden_layer1_size, 
                      hidden_layer2_size, out_layer_size,
                      X, y, lmbd)

    a.learn()
    a.check()

nn_train()


def checkGrad():
    
    def debugInit(fan_out, fan_in):
        W = np.zeros((fan_out, 1 + fan_in))
        W = np.sin(1 + np.arange(W.size)) / 10
        return W

    input_layer_size = 3
    hidden_layer1_size = 5
    hidden_layer2_size = 5
    out_layer_size= 3
    m = 5
    lmbd = 0

    # We generate some 'random' test data
    Theta1 = debugInit(hidden_layer1_size, input_layer_size).reshape(
        hidden_layer1_size, 1+input_layer_size)
    Theta2 = debugInit(hidden_layer2_size, hidden_layer1_size).reshape(
        hidden_layer2_size, 1+hidden_layer1_size)
    Theta3 = debugInit(out_layer_size, hidden_layer2_size).reshape(
        out_layer_size, 1+hidden_layer2_size)
    nn_params = np.hstack((Theta1.ravel(), Theta2.ravel(), Theta3.ravel()))

    # Reusing debugInitializeWeights to generate X
    X  = debugInit(m, input_layer_size - 1).reshape(m, input_layer_size, order='F')
    y  = np.array([[1,2,0,1,2]]).T

    a = neuralNetwork(nn_params, hidden_layer1_size, hidden_layer2_size,
                      out_layer_size, X, y, lmbd)

    #print "Theta1: \n", Theta1
    #print "Theta2 \n", Theta2
    #print "X: \n", X
    #print "nn_params: \n", nn_params
    print "Grad: \n", a.nnGrad(nn_params)
    print "Numerical grad: \n", a.computeNumGradient()

#checkGrad()

def learnKaggleData():
    X = np.loadtxt('test.csv', delimiter=',')
    y = np.loadtxt('y_out.csv', delimiter=',')
    #y = np.zeros(28000)

    input_layer_size = 784
    hidden_layer1_size = 250
    hidden_layer2_size = 250
    out_layer_size = 10
    lmbd = 0

    Theta1 = np.loadtxt('T1_out.csv', delimiter=',')
    Theta2 = np.loadtxt('T2_out.csv', delimiter=',')
    Theta3 = np.loadtxt('T3_out.csv', delimiter=',')
    nn_params = np.hstack((Theta1.ravel(order='F'), 
                           Theta2.ravel(order='F'), 
                           Theta3.ravel(order='F')))

    a = neuralNetwork(nn_params, hidden_layer1_size, hidden_layer2_size,
                      out_layer_size, X, y, lmbd)

    #a.check()
    a.look()

#learnKaggleData()
