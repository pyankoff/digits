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

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


raw_image = misc.imread('page_3.png', flatten=1)

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
hidden_layer1_size = 200
hidden_layer2_size = 200
out_layer_size = 10


W1 = np.loadtxt('output/W1.csv', delimiter=',')
W2 = np.loadtxt('output/W2.csv', delimiter=',')
b1 = np.loadtxt('output/b1.csv', delimiter=',')
b2 = np.loadtxt('output/b2.csv', delimiter=',')
softmaxTheta = np.loadtxt('output/softmax.csv', delimiter=',')

def forwardprop(data):
    z2 = np.dot(W1, data) + b1
    a2 = sigmoid(z2)

    z3 = np.dot(W2, a2) + b2
    a3 = sigmoid(z3)

    z4 = np.dot(softmaxTheta, a3)
    h = np.exp(z4) / np.sum(np.exp(z4), axis=0)
    #result = np.argmax(h, axis=0)
    print h
    return h

"""
%3x5   3x4         4x5
z2 = stack{1}.w * data + repmat(stack{1}.b, 1, columns(data));
a2 = sigmoid(z2);

%5x5  5x3        3x5
z3 = stack{2}.w * a2 + repmat(stack{2}.b, 1, columns(a2));
a3 = sigmoid(z3);

%2x5     2x5        5x5
z4 = softmaxTheta * a3;
h = exp(z4);

cost = -sum(sum(groundTruth .* ...
    log(h./repmat(sum(h, 1), rows(h), 1)))) ./ M + ...
    0.5 * lambda * sum(sum(softmaxTheta.^2));;
"""



res = [0] * (len(cuts) - 1)

for i in xrange(len(cuts)-1):
    digit = 255 - cut_image[:,cuts[i]:cuts[i+1]]

    h, w = digit.shape

    t1 = identity(28, h)
    t2 = identity(w, 28)

    resized_digit = np.dot(t1, np.dot(digit, t2))
    resized_digit = np.where(resized_digit < np.mean(resized_digit) - np.std(resized_digit), 0, 1)
    resized_digit = 1 - resized_digit


    plt.imshow(resized_digit, cmap=plt.cm.gray)
    plt.show()


    res[i] = str(np.argmax(forwardprop(resized_digit.ravel())))

    print "Recognized digit: ", res[i]
    wait = raw_input("press enter to continue")

print '  ', '  '.join(res)
