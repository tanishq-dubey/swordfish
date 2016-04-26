import numpy as np
import time

learningRate = 0.2

def sigmoid(z):
	return np.divide(1.0, (1+np.exp(-z)))

def dSigmoid(z):
	return sigmoid(z) * (1 - sigmoid(z))

def error(t, o):
	return .5 * np.power((t-o), 2)

def dError(t, o):
	return (o - t)

class layer:
	
	def __init__(self, numNodes, bias, inVals):
		self.numNodes = numNodes
		self.bias = bias
		self.inVals = inVals
		self.weights = np.random.uniform(0,1,size=(numNodes, len(inVals)))
		self.weightDelta = np.random.uniform(0,0,size=(numNodes, len(inVals)))
		self.layerNet = np.zeros(numNodes)
		self.layerOut = np.zeros(numNodes)
		
	def evaluate(self):
		self.calculateSum()
		self.calculateSigmoid()

	def calculateSigmoid(self):
		for x in range(len(self.layerNet)):
			self.layerOut[x] = sigmoid(self.layerNet[x])

	def calculateSum(self):
		for x in range(self.numNodes):
			self.layerNet[x] = np.dot(self.inVals, np.transpose(self.weights[x])) + self.bias

def layerTwoBackprop(l, t):
	for x in range(len(l.weights)): #Should be number of out nodes
		for y in range(len(l.weights[x])): #Should be number of input nodes
			l.weightDelta[x][y] = dSigmoid(l.layerOut[x]) * dError(t, l.layerOut[x])
			dW = l.weightDelta[x][y] * l.inVals[y]
			l.weights[x][y] = l.weights[x][y] - (learningRate * dW)

def layerOneBackprop(l, a):
	for x in range(len(l.weights)):
		for y in range(len(l.weights[x])):
			dW = a.weightDelta[0][x] * l.inVals[y] * a.weights[0][x] * dSigmoid(l.layerOut[x])
			l.weights[x][y] = l.weights[x][y] - (learningRate * dW)

#Parse the File data
# Open the file and put it in a list of lines
f = open('mushroom-training.txt', 'r').readlines()
runNum = 0
lineNum = 0

#For testing, grab the first line
#	Should be going through all lines
l = f[lineNum]
# Remove whitespace, split by comma
p = l.replace(' ','').split(',')
# Convert the list from strings to ints
d = map(int, p)
# Grab the target, leaving only attributes to program off of
t = d.pop(0)
x = layer(3, np.random.uniform(-1,1), d)
y = layer(1, np.random.uniform(-1,1), x.layerOut)
errorSum = 0

while runNum < 50001:
	l = f[lineNum]
	# Remove whitespace, split by comma
	p = l.replace(' ','').split(',')
	# Convert the list from strings to ints
	d = map(int, p)
	# Grab the target, leaving only attributes to program off of
	t = d.pop(0)
	x.inVals = d
	x.evaluate()
	y.inVals = x.layerOut
	y.evaluate()
	layerTwoBackprop(y, t)
	layerOneBackprop(x, y)
	currError = error(t, y.layerOut[0])
	errorSum = errorSum + currError
	if runNum % 1000 == 0:
		np.save('weights1.npy', x.weights)
		np.save('weights2.npy', y.weights)
		np.savetxt('weights1.txt', x.weights)
		np.savetxt('weights2.txt', y.weights)
		print "RUN NUMBER: "+ str(runNum) +" LOSS: " + str(currError)
		print "TARGET VAL VS OUTPUT: " + str(t) + ", " + str(y.layerOut[0])
		print "ERR AVG: " + str(errorSum/runNum)
		print y.weights
		print ""
	runNum = runNum + 1
	lineNum = lineNum + 1
	if lineNum >= len(f):
		lineNum = 0

print "SWITCHING TO TEST DATA"
print "======================"
raw_input()

f = open('mushroom-testing.txt', 'r').readlines()
runNum = 0
lineNum = 0

while True:
	l = f[lineNum]
	# Remove whitespace, split by comma
	p = l.replace(' ','').split(',')
	# Convert the list from strings to ints
	d = map(int, p)
	# Grab the target, leaving only attributes to program off of
	t = d.pop(0)
	x.inVals = d
	x.evaluate()
	y.inVals = x.layerOut
	y.evaluate()
	#layerTwoBackprop(y, t)
	#layerOneBackprop(x, y)
	currError = error(t, y.layerOut[0])
	errorSum = errorSum + currError
	if runNum % 100 == 0:
		np.save('weights1.npy', x.weights)
		np.save('weights2.npy', y.weights)
		np.savetxt('weights1.txt', x.weights)
		np.savetxt('weights2.txt', y.weights)
		print "RUN NUMBER: "+ str(runNum) +" LOSS: " + str(currError)
		print "TARGET VAL VS OUTPUT: " + str(t) + ", " + str(y.layerOut[0])
		print "ERR AVG: " + str(errorSum/runNum)
		print y.weights
		print ""
	runNum = runNum + 1
	lineNum = lineNum + 1
	if lineNum >= len(f):
		lineNum = 0
