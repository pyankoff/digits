import numpy as np
from learning import forwardProp, nnCostFunction


def debugInit(fan_out, fan_in):
    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.sin(np.arange(W.size)).reshape(W.shape)
    return W


def computeNumGradient(theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4

    for p in range(theta.shape[0]):
        # Set perturbation vector
        perturb[p] = e
        loss1 = nnCostFunction(theta - perturb,
                               input_layer_size,
                               hidden_layer_size,
                               out_layer_size,
                               X, y, lmbd)
        loss2 = nnCostFunction(theta + perturb,
                               input_layer_size,
                               hidden_layer_size,
                               out_layer_size,
                               X, y, lmbd)

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;

    return numgrad


input_layer_size = 3
hidden_layer_size = 5
out_layer_size= 3
m = 5
lmbd = 0

# We generate some 'random' test data
Theta1 = debugInit(hidden_layer_size, input_layer_size)
Theta2 = debugInit(out_layer_size, hidden_layer_size)
nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))

# Reusing debugInitializeWeights to generate X
X  = debugInit(m, input_layer_size - 1)
y  = np.array([[1,2,0,1,2]]).T

print computeNumGradient(nn_params)