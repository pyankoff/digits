# recognition from line of digits

import numpy as np
from learning_class_2hidden import neuralNetwork
from scipy import misc
import matplotlib.pyplot as plt
from processing import clearing
from scipy import ndimage


def identity(m, n):
    t = np.zeros((m, n))
    for i in xrange(n):
        for j in xrange(m):
            if int(i * m / n) == j:
                t[j,i] = 1
            else:
                t[j,i] = 0
    return t

raw_image = misc.imread('7.png', flatten=1)

print "raw_image"
plt.imshow(raw_image, cmap=plt.cm.gray)
plt.show()

cut_image = clearing(raw_image)
#print "cut_image"
#plt.imshow(cut_image, cmap=plt.cm.gray)
#plt.show()



#raw_image = misc.imread('line2.png', flatten=1)

#plt.imshow(raw_image, cmap=plt.cm.gray)
#plt.show()

cut_image = np.where(cut_image < np.mean(cut_image), 0, 255)

h, w = cut_image.shape

hist = np.sum(cut_image, axis=0)
plt.plot(hist)
plt.show()

cuts = []
position = 0
zeros = 0

for i in range(w):
    if hist[i] == 0:  
        position += i
        zeros += 1
    elif zeros:
        cuts.append(position/zeros)
        position = 0
        zeros = 0
cuts.append(w)

print cuts


input_layer_size = 784
hidden_layer1_size = 250
hidden_layer2_size = 250
out_layer_size = 10
lmbd = 0.0001

Theta1 = np.loadtxt('T1_out.csv', delimiter=',')
Theta2 = np.loadtxt('T2_out.csv', delimiter=',')
Theta3 = np.loadtxt('T3_out.csv', delimiter=',')
nn_params = np.hstack((Theta1.ravel(order='F'), 
                       Theta2.ravel(order='F'),
                       Theta3.ravel(order='F')))

X = np.loadtxt('X_5000.csv', delimiter=',')
y = np.loadtxt('y_5000.csv', delimiter=',')

a = neuralNetwork(nn_params, hidden_layer1_size, hidden_layer2_size,
                  out_layer_size, X, y, lmbd)

res = [0] * (len(cuts) - 1)

for i in xrange(len(cuts)-1):
    digit = 255 - cut_image[:,cuts[i]:cuts[i+1]]

    h, w = digit.shape

    t1 = identity(28, h)
    t2 = identity(w, 28)

    resized_digit = np.dot(t1, np.dot(digit, t2))
    resized_digit = np.where(resized_digit < np.mean(resized_digit) - np.std(resized_digit), 0, 255)
    resized_digit = 255 - resized_digit


    plt.imshow(resized_digit, cmap=plt.cm.gray)
    plt.show()


    res[i] = str(np.argmax(a.forwardProp(Theta1, Theta2, Theta3, 
                 resized_digit.reshape((1, input_layer_size)))[0]))

    print "Recognized digit: ", res[i]
    wait = raw_input("press enter to continue")

print '  ', '  '.join(res)
