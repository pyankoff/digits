import numpy as np
from scipy import optimize
from math import log
import matplotlib.pyplot as plt

class neuralNetwork(object):

    def __init__(self, nn_params, hidden_layer1_size, hidden_layer2_size,
                 out_layer_size, X, y, lmbd):
        self.nn_params = nn_params
        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.out_layer_size = out_layer_size
        self.X = X
        self.y = y
        self.lmbd = lmbd
        self.m, self.input_layer_size = X.shape                
        self.yv = np.zeros((self.m, self.out_layer_size))
        for k in range(self.m):
            self.yv[k, int(y[k])] = 1

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoidGradient(self, z):

        g = np.zeros(z.shape)
        g = self.sigmoid(z) * (1 - self.sigmoid(z))
        return g

    def params_to_theta(self, nn_params):
        Theta1 = nn_params[0:(self.hidden_layer1_size * 
                          (self.input_layer_size + 1))].reshape(
                          self.hidden_layer1_size, (self.input_layer_size + 1), order='F')
        
        Theta2 = nn_params[(self.hidden_layer1_size * 
                           (self.input_layer_size + 1)):-self.out_layer_size * 
                           (self.hidden_layer2_size + 1)].reshape(
                            self.hidden_layer2_size, (self.hidden_layer1_size + 1), order='F')

        Theta3 = nn_params[-self.out_layer_size * 
                           (self.hidden_layer2_size + 1):].reshape(
                           self.out_layer_size, (self.hidden_layer2_size + 1), order='F')
        
        Theta1reg = Theta1.copy()
        Theta1reg[:, 0] = np.zeros((1, self.hidden_layer1_size))
        Theta2reg = Theta2.copy()
        Theta2reg[:, 0] = np.zeros((1, self.hidden_layer2_size))
        Theta3reg = Theta3.copy()
        Theta3reg[:, 0] = np.zeros((1, self.out_layer_size))
        return Theta1, Theta2, Theta3, Theta1reg, Theta2reg, Theta3reg

    def forwardProp(self, Theta1, Theta2, Theta3, Xrows):

        a1 = np.hstack((np.ones((Xrows.shape[0], 1)), Xrows))
        z2 = np.dot(a1, Theta1.T)
        a2 = np.hstack((np.ones((Xrows.shape[0], 1)), self.sigmoid(z2)))
        z3 = np.dot(a2, Theta2.T)
        a3 = np.hstack((np.ones((Xrows.shape[0], 1)), self.sigmoid(z3)))
        z4 = np.dot(a3, Theta3.T)
        a4 = self.sigmoid(z4)

        return a4, z3, a3, z2, a2, a1

    def nnCostFunction(self, nn_params):

        Theta1, Theta2, Theta3, Theta1reg, Theta2reg, Theta3reg = self.params_to_theta(nn_params)

        a4 = self.forwardProp(Theta1, Theta2, Theta3, self.X)[0]

        J = 0

        for i in range(self.out_layer_size):
            cost0 = np.dot((1 - self.yv[:, i]).T, np.log(1 - a4[:, i]))
            cost1 = np.dot(self.yv[:, i].T, np.log(a4[:, i]))
            J = J - 1 * (cost0 + cost1) / self.m

        #regularization
        t1reg = (Theta1reg ** 2).sum()
        t2reg = (Theta2reg ** 2).sum()
        t3reg = (Theta3reg ** 2).sum()

        reg = 0.5 * self.lmbd * (t1reg + t2reg + t3reg) / self.m

        J = J + reg

        print J
        return J

    def nnGrad(self, nn_params):

        # BACK PROPAGATION
        Theta1, Theta2, Theta3, Theta1reg, Theta2reg, Theta3reg = self.params_to_theta(nn_params)

        D1 = np.zeros(Theta1.shape)
        D2 = np.zeros(Theta2.shape)
        D3 = np.zeros(Theta3.shape)

        for t in xrange(self.m):
            # step 1
            a4, z3, a3, z2, a2, a1 = self.forwardProp(Theta1, Theta2, Theta3,
                self.X[t,:].reshape((1, self.input_layer_size)))
            # step 2
            delta4 = (a4 - self.yv[t, :]).reshape(self.out_layer_size,1)         
            # step 3
            delta3 = np.dot(Theta3[:, 1:].T, delta4) * self.sigmoidGradient(z3.T)
            # step 4
            delta2 = np.dot(Theta2[:, 1:].T, delta3) * self.sigmoidGradient(z2.T)
            # step 5
            D1 = D1 + np.dot(delta2, a1)
            D2 = D2 + np.dot(delta3, a2)
            D3 = D3 + np.dot(delta4, a3)


        Theta1_grad = D1 / self.m + self.lmbd * Theta1reg / self.m
        Theta2_grad = D2 / self.m + self.lmbd * Theta2reg / self.m 
        Theta3_grad = D3 / self.m + self.lmbd * Theta3reg / self.m

        grad = np.hstack((Theta1_grad.ravel(order='F'), 
                          Theta2_grad.ravel(order='F'),
                          Theta3_grad.ravel(order='F')))

        return grad

    def computeNumGradient(self):
        numgrad = np.zeros(self.nn_params.shape)
        perturb = np.zeros(self.nn_params.shape)
        e = 1e-4

        for p in range(self.nn_params.size):
            # Set perturbation vector
            perturb[p] = e
            loss1 = self.nnCostFunction(self.nn_params - perturb)
            loss2 = self.nnCostFunction(self.nn_params + perturb)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e);
            perturb[p] = 0;

        return numgrad


    def learn(self):

        self.nn_params = optimize.fmin_cg(self.nnCostFunction, self.nn_params, 
                                fprime=self.nnGrad, retall=0,
                                maxiter=500)

        Theta1, Theta2, Theta3, Theta1reg, Theta2reg, Theta3reg = self.params_to_theta(self.nn_params)

        np.savetxt('T1_out.csv', Theta1, delimiter=',')
        np.savetxt('T2_out.csv', Theta2, delimiter=',')
        np.savetxt('T3_out.csv', Theta3, delimiter=',')
        


    def check(self):
        Theta1, Theta2, Theta3, Theta1reg, Theta2reg, Theta3reg = self.params_to_theta(self.nn_params)

        a4 = self.forwardProp(Theta1, Theta2, Theta3, self.X)[0]
        y_res = np.argmax(a4, axis=1)
        np.savetxt('y_out.csv', y_res, delimiter=',')
        #print (self.y==y_res).sum()/self.m
        print float((y_res==self.y).sum())/self.m


    def look(self):
        Theta1, Theta2, Theta3, Theta1reg, Theta2reg, Theta3reg = self.params_to_theta(self.nn_params)

        while raw_input() != 'q':
            i = int(np.random.random(1) * 5000)
            res = np.argmax(self.forwardProp(Theta1, Theta2, Theta3, self.X[i,:].reshape(
                    (1, self.input_layer_size)))[0])
            plt.imshow(self.X[i,:].reshape((28,28)), cmap=plt.cm.gray)
            plt.title("%d" % res)
            plt.show()



