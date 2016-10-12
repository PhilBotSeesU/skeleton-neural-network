import numpy
import scipy.special

class NeuralNetwork:

    #Create neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        #Link weight matrice created -
        #wih = weight links from input layer to hidden layer
        #who = weight links from hidden layer to output layer
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), self.hnodes, self.inodes)
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), self.onodes, self.hnodes)

        #Learning rate
        self.lr = learningRate

        #Activate func = sigmoid func
        self.activationFunction = lambda x: scipy.special.expit(x)

        pass

    #neural net training
    def train(self, inputList, targetList):

        #Flatten inputs to 2D array
        inputs = numpy.array(inputList, ndmin=2).T
        targets = numpy.array(targetList, ndmin=2).T

        #Calculate szignals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        #Calc signals emerging from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)

        #Calc signals into final output layer
        finalInputs = numpy.dot(self.wih, hiddenOutputs)
        #Calc signals emerging from final output layer
        finalOutputs = self.activationFunction(finalOutputs)

        #Output layuer error = (target - actual)
        outputErrors = target - finalOutputs
        #hidden layer error is the outpuyt errors split by weights
        #ratioed and recombine dby hidden nodes
        hiddenErrors = numpy.dot(self.who.T, outputErrors)

        #Update wseights for links between the hidden and output layers
        self.who += self.lr * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)),
        numpy.transpose(hiddenOutputs))

        #Update weights for links between input and hidden, working our way back to initial layer
        self.whi += self.lr * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)),
        numpy.transpose(inputs))

        pass

    #Query neural network
    def query(self, inputLists):
        #Flatten input to 2D
        inputs = numpy.array(inputLists, ndmin=2).T

        #Calc signalks into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        #Calc signals from emerging layer
        hiddenOutputs = self.activationFunction(hiddenInputs)

        #Calc signals into final output layer
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        #Calc signals emerging from final layer
        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs
