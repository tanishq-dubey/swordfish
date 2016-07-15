import numpy as np
import os, struct
from array import array as pyarray
import time

learningRate = 0.2

weightInitMin = -1
weightInitMax = 1

biasInitMin = 0
biasInitMax = 1

# Source: http://g.sweyla.com/blog/2012/mnist-numpy/
def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

class layer:

    def __init__(self, numberNodes, inVals, layerAhead=None, useBias=False, isOutput=False):
        self.numberNodes = numberNodes
        self.inVals     = inVals
        # Initalize weights randomly (set min and max to same value to initialize with constant)
        self.weights    = np.random.uniform(weightInitMin, weightInitMax, size=(len(inVals), numberNodes))
        # Deltas are used when backpropagating future layers, saves lots of computation
        self.delta      = np.zeros((len(inVals), numberNodes))
        # If biases are being used (to avoid local minima) initalize randomly here
        self.bias       = np.random.uniform(biasInitMin, biasInitMax, size=(1, numberNodes))
        self.biasWeight = np.random.uniform(weightInitMin, weightInitMax, size=(1, numberNodes))
        self.net        = np.zeros(numberNodes)
        self.out        = np.zeros(numberNodes)
        # Only used for hidden layers
        self.layerAhead = layerAhead
        self.isOutput   = isOutput
        self.useBias    = useBias

    # Sum up and normalize for regualar Feedforward NN
    def forward(self):
        self.calculateSum()
        self.normalize()

    def back(self, targetVals):
        if(self.isOutput):
            # Find the values for the deltas...
            e = np.vectorize(self.dError)
            eOut = e(targetVals, self.out)
            s = np.vectorize(self.dSigmoidFast)
            sOut = s(self.out)
            # ...Then store them
            self.delta = np.multiply(eOut, sOut)
            for x in range(len(self.inVals)):
                # Calculate change for each set of weights
                dW = np.multiply(self.delta, self.inVals[0][x])
                # The new weight is the delta multiplied by the learning rate, to avoid overstepping
                self.weights[x] = np.subtract(self.weights[x], np.multiply(dW, learningRate))
        else:
            # Calculate sum of layer ahead's deltas and weights
            x = np.sum(np.multiply(self.layerAhead.delta, self.layerAhead.weights), axis = 1)
            s = np.vectorize(self.dSigmoidFast)
            sOut = s(self.out)
            self.delta = np.multiply(x, sOut)
            for x in range(len(self.inVals)):
                # Calculate change for each set of weights
                dW = np.multiply(self.delta, self.inVals[x])
                # The new weight is the delta multiplied by the learning rate, to avoid overstepping
                self.weights[x] = np.subtract(self.weights[x], np.multiply(dW, learningRate))

    # Dots the weights and inputs together, optionally adds biases
    def calculateSum(self):
        self.net = np.dot(self.inVals, self.weights)
        if(self.useBias):
            x = np.multiply(self.bias, self.biasWeight)
            self.net = np.add(self.net, x)

    # Simply the net values through the sigmoid
    def normalize(self):
        x = np.vectorize(self.sigmoid)
        self.out = x(self.net)

    # Logistic sigmoid
    def sigmoid(self, x):
        return np.divide(1.0, (1 + np.exp(-x)))

    # Trick for using the precomputed output values
    def dSigmoidFast(self, x):
        return (x * (1 - x))

    # If you actually ever need to calculate the d-Sigmoid of a value
    def dSigmoid(self, x):
        return (self.sigmoid(x) * (1 - self.sigmoid(x)))

    # Derivative of RMS error
    def dError(self, t, o):
        return (o - t)

# BEGIN PROGRAM
# ================================================

# Load MNIST training data
images, labels = load_mnist(path="./trainingData/")
# Prevent overflows
images = np.divide(images, 255.0)

# Create Layers
hiddenOne = layer(28, images[0].flatten(), None, True, False)
hiddenTwo = layer(28, hiddenOne.out, None, True, False)
outLayer  = layer(10, hiddenOne.out, None, True, True)

# Define the order of layers
hiddenOne.layerAhead = hiddenTwo
hiddenTwo.layerAhead = outLayer

# Other rumtime variables
epochNumber = 1
currentError = 100
# Used to calculate average error
errorSum = 0
avgError = 100
numberCorrect = 0
numberWrong = 0

while True:
    # Set input values and forward pass
    hiddenOne.inVals = images[epochNumber % len(images)].flatten()
    hiddenOne.forward()
    hiddenTwo.inVals = hiddenOne.out
    hiddenTwo.forward()
    outLayer.inVals = hiddenTwo.out
    outLayer.forward()
    # Construct target vector
    targetValue = labels[epochNumber % len(images)][0]
    targetVector = np.zeros((1,10))
    targetVector[0][targetValue] = 1
    # Calculate error
    currentError = (np.sum(np.multiply(np.square(np.subtract(targetVector, outLayer.out)), 0.5)) * 100)
    errorSum = errorSum + currentError
    avgError = errorSum / epochNumber
    outputValue = np.argmax(outLayer.out)
    if(outputValue == targetValue):
        numberCorrect+= 1
    else:
        numberWrong += 1
    # Print target, actual, and error percent every 2,000 epochs
    if(epochNumber % 2000 == 0) or (epochNumber == 1):
        print("RUN NUMBER:        " + str(epochNumber))
        print("CURRENT RUN ERROR: " + str(currentError) + "%")
        print("TARGET VECTOR:     " + np.array2string(targetVector).replace('\n', ''))
        print("OUTPUT VECTOR:     " + np.array2string(outLayer.out).replace('\n', ''))
        print("AVERAGE ERROR:     " + str(errorSum/2000) + "%")
        print("TARGET VALUE:      " + str(targetValue))
        print("OUTPUT VALUE:      " + str(outputValue))
        print("NUM CORRECT/WRONG: " + str(numberCorrect) + "/" + str(numberWrong))
        print("ABS ERROR:         " + str((np.abs(epochNumber - numberCorrect)/epochNumber)*100) + "%")
        print("")
        errorSum = 0
    # Backprop
    outLayer.back(targetVector)
    hiddenTwo.back(targetVector)
    hiddenOne.back(targetVector)
    # Update runtime values
    epochNumber = epochNumber + 1
    # Loop


print("RUN NUMBER:        " + str(epochNumber))
print("CURRENT RUN ERROR: " + str(currentError))
print("TARGET VECTOR:     " + str(targetVector))
print("OUTPUT VECTOR:     " + str(outLayer.out))
print("AVERAGE ERROR:     " + str(avgError))
