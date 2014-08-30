#!/usr/bin/python

from random import uniform
import math

class SNeuron():
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.vec_weight = [ uniform(-1,1) for x in range(0,num_inputs+1) ]

class SNeuronLayer():
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.num_neurons = num_neurons
        self.vec_neurons = [ SNeuron(num_inputs_per_neuron) for x in range(0,num_neurons) ]


class CNeuralNet():
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, neurons_per_hidden_layer):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers

        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.vec_layers = []
        num_neurons_previous_layer = num_inputs
        num_neurons = neurons_per_hidden_layer
        for i in range(0, num_hidden_layers):
            self.vec_layers.append(SNeuronLayer(num_neurons, num_neurons_previous_layer))
            num_neurons_previous_layer = neurons_per_hidden_layer
        num_neurons = num_outputs
        self.vec_layers.append(SNeuronLayer(num_neurons, num_neurons_previous_layer))


    def createNet():
        print "createNet"

    def getWeights():
        print "getWeights"

    def getNumberOfWeights():
        print "getNumberOfWeights"

    def putWeights(self, weights):
        print "putWeights"

    def update(self, inputs):
        print "update"
        # stores the resultant outputs from each layer
        # vector<double> outputs;
        outputs = []
        c_weight = 0

        # first check that we have the correct amount of inputs
        if len(inputs) != self.num_inputs:
            # just return an empty vector if incorrect.
            return outputs

        # For each layer....
        # for(i=0; i<self.num_hidden_layers + 1; ++i){
        for i in range(0, self.num_hidden_layers + 1):
            if i > 0:
                inputs = outputs
            outputs = []
            c_weight = 0

            # for each neuron sum the (inputs * corresponding weights).Throw
            # the total at our sigmoid function to get the output.
            # for (int j=0; j<m_vecLayers[i].m_NumNeurons; ++j)
            for j in range(0, self.vec_layers[i].num_neurons):

                netinput = 0.0

                num_inputs = self.vec_layers[i].vec_neurons[j].num_inputs

                # for each weight
                # for (int k=0; k<NumInputs - 1; ++k)
                for k in range(0, num_inputs):
                    # sum the weights x inputs
                    netinput += self.vec_layers[i].vec_neurons[j].vec_weight[k] * inputs[c_weight]
                    c_weight += 1

                # add in the bias
                netinput += self.vec_layers[i].vec_neurons[j].vec_weight[num_inputs-1] * -1
                # netinput += m_vecLayers[i].m_vecNeurons[j].m_vecWeight[NumInputs-1] * -1;

                # we can store the outputs from each layer as we generate them.
                # The combined activation is first filtered through the sigmoid
                # function
                # outputs.push_back(Sigmoid(netinput, CParams::dActivationResponse));
                outputs.append(self.sigmoid(netinput, 1))
                # outputs.append(netinput)

                c_weight = 0
            print outputs
        return outputs

    def sigmoid(self, activation, response):
        # print "sigmoid"
        return 1 / (1 + math.exp(-(activation)));



if __name__ == '__main__':
    nn = CNeuralNet(2,2,2,6)
    # print nn.vec_layers
    layer = 0
    for vec_layer in nn.vec_layers:
        layer += 1
        print "layer %d" %layer
        layer_neuron = 0
        for neuron in vec_layer.vec_neurons:
            layer_neuron += 1
            print "neuron %d" %layer_neuron
            print neuron.vec_weight

    print nn.update([1,1])
