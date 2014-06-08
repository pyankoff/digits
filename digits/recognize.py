import numpy as np
from learning_class import neuralNetwork
from scipy import misc
import matplotlib.pyplot as plt

np.seterr(over='ignore')

def recognizeDigit():
    input_layer_size = 784
    hidden_layer_size = 25
    out_layer_size = 10
    lmbd = 0.1

    Theta1 = np.loadtxt('digits/T1_out.csv', delimiter=',')
    Theta2 = np.loadtxt('digits/T2_out.csv', delimiter=',')
    nn_params = np.hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

    X = np.loadtxt('digits/X_5000.csv', delimiter=',')
    y = np.loadtxt('digits/y_5000.csv', delimiter=',')

    a = neuralNetwork(nn_params, hidden_layer_size, out_layer_size,
                      X, y, lmbd)


    filename = 'uploads/4.PNG'
    raw_image = misc.imread(filename, flatten=1)
    plt.imshow(raw_image, cmap=plt.cm.gray)
    raw_image = np.where(raw_image < 0.9*np.mean(raw_image) - 
                         np.std(raw_image), 0, 255)
    #plt.imshow(raw_image, cmap=plt.cm.gray)
    im_m, im_n = raw_image.shape

    t1 = np.zeros((28, im_m))
    t2 = np.zeros((im_n, 28))

    # 145x145 to 28x28 transition array
    for i in range(im_m):
        for j in range(28):
            if int(i * 28 / im_m) == j:
                t1[j,i] = 1
            else:
                t1[j,i] = 0

    # 145x145 to 28x28 transition array
    for i in range(im_n):
        for j in range(28):
            if int(i * 28 / im_n) == j:
                t2[i,j] = 1
            else:
                t2[i,j] = 0

    b = np.dot(t1, np.dot(raw_image, t2))
    #plt.imshow(b, cmap=plt.cm.gray)
    b = np.where(b < np.mean(b) - 1*np.std(b), 0, 255)
    #plt.imshow(b, cmap=plt.cm.gray)
    image = 255 - b

    res = np.argmax(a.forwardProp(Theta1, Theta2, image.reshape(
                (1, input_layer_size)))[0])

    plt.show()
    print res


recognizeDigit()
