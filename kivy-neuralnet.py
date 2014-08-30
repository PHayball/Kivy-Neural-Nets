#!/usr/bin/python
import kivy
kivy.require('1.8.0') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.graphics.instructions import Instruction 
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from random import uniform, randint
from kivy.clock import Clock
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
        return outputs

    def sigmoid(self, activation, response):
        # print "sigmoid"
        return 1 / (1 + math.exp(-(activation)));

class NeuronCell(Widget):
    neuron = ObjectProperty(None)
    value = NumericProperty(0)
    layer = StringProperty('hidden')
    # label = StringProperty(False)
    # def __init__(self):
    #   with self.canvas:
    #       Ellipse(pos=self.pos, size=self.size)
    def load_widget(self, grid_x, grid_y):
        (self.grid_x, self.grid_y) = (grid_x, grid_y)
        (real_x, real_y) = (self.parent.grid_width*grid_x, self.parent.height - (self.parent.grid_height*grid_y))
        # self.label = Label(center=self.parent.center, text="%s" %self.neuron)

        with self.canvas:
            Ellipse(pos=(real_x-5, real_y-5), size=(10, 10))

        if self.layer == 'input':
            self.label = Label(center=(real_x, real_y + 20), text="-")
            self.add_widget(self.label)
        elif self.layer == 'output':
            self.label = Label(center=(real_x, real_y - 20), text="-")
            self.add_widget(self.label)

    def update(self, value=False):
        if value and self.label:
            self.label.text = value
    pass    

class NeuralDisplay(Widget):
    # neural_network = ObjectProperty(0)
    # n = NeuronCell()
    def load_content(self):
        with self.canvas:
            for i in range(10):
                Ellipse(pos=(20*i, 10), size=(10, 10))


    def linkDisplayToNetwork(self):

        self.layers = []
        ## Link Inputs
        print "linking input layer"
        tmp = []
        row = 1
        grid_x_arr = self.getLayerPositions(self.neural_network.num_inputs, self.num_columns)
        grid_y = row

        for i in range(self.neural_network.num_inputs):
            grid_x = grid_x_arr[i]
            neuron_wid = NeuronCell(layer='input')
            self.add_widget(neuron_wid)
            neuron_wid.load_widget(grid_x, grid_y)
            tmp.append(neuron_wid)
        self.layers.append(tmp)

        ## Link Hidden
        print "linking hidden layer(s)"
        for layer in self.neural_network.vec_layers[:-1]:
            tmp = []
            row += 1
            grid_x_arr = self.getLayerPositions(self.neural_network.neurons_per_hidden_layer, self.num_columns)
            grid_y = row
            i=0
            for neuron in layer.vec_neurons:
                grid_x = grid_x_arr[i]
                i+=1
                neuron_wid = NeuronCell(neuron=neuron, layer='hidden')
                self.add_widget(neuron_wid)
                neuron_wid.load_widget(grid_x, grid_y)
                tmp.append(neuron_wid)
            self.layers.append(tmp)

        ## Link Outputs
        print "linking output layer"
        tmp = []
        row = self.num_rows
        grid_x_arr = self.getLayerPositions(self.neural_network.num_outputs, self.num_columns)
        grid_y = row
        i=0
        for neuron in self.neural_network.vec_layers[-1].vec_neurons:
            grid_x = grid_x_arr[i]
            i+=1
            neuron_wid = NeuronCell(neuron=neuron, layer='output')
            self.add_widget(neuron_wid)
            neuron_wid.load_widget(grid_x, grid_y)
            tmp.append(neuron_wid)
        self.layers.append(tmp)

    def defineGridPositions(self):
        self.grid_height = self.height / (self.num_rows + 1)
        self.grid_width = self.width / (self.num_columns + 1)
        grid_positions = []

        # Input Layer
        grid_positions.append(self.getLayerPositions(self.neural_network.num_inputs, self.num_columns))

        # Hidden Layers
        for i in range(self.neural_network.num_hidden_layers):
            grid_positions.append(self.getLayerPositions(self.neural_network.neurons_per_hidden_layer, self.num_columns))

        # Output Layer
        grid_positions.append(self.getLayerPositions(self.neural_network.num_outputs, self.num_columns))

        self.grid_positions = grid_positions

    def getLayerPositions(self, n, m):
        if n > m:
            return []

        diff = m-n
        start_x = 1 + (diff / 2)

        if diff % 2:
            start_x += 0.5

        return [ start_x + i for i in range(n) ]
    def getLayerPositions(self, n, m):
        if n > m:
            return []

        diff = m-n
        start_x = 1 + (diff / 2)

        if diff % 2:
            start_x += 0.5

        return [ start_x + i for i in range(n) ]


    def update(self, inputs=False):
        # inputs = [1,0]
        inputs = [ randint(0,1) for x in range(self.neural_network.num_inputs) ]
        outputs = self.neural_network.update(inputs)

        i=0
        for neuron_wid in self.layers[0]:
            neuron_wid.update(str(inputs[i]))
            i+=1
        i=0
        for neuron_wid in self.layers[-1]:
            neuron_wid.update(str(outputs[i]))
            i+=1

        # self.draw(inputs, outputs)
        # with self.canvas:
        #     # Label(pos=self.center, text="%s" %self.num_columns)
        #     # Label(pos=self.center, text="%s" %inputs[0])
        #     if outputs:
        #         i = 0
        #         for val in inputs:
        #             Label(pos=(self.grid_width*self.grid_positions[0][i], self.height - (self.grid_height)), text=str(val))
        #             i += 1


    def draw(self, inputs=False, outputs=False):
        print self.layers
        # with self.canvas:
            # Label(pos=self.center, text="%s" %self.num_columns)

            # Label(center=self.center, text="%s" %inputs)
                # row_num = 0
                # for row in self.grid_positions:
                #     row_num += 1
                #     i = 0
                #     for position in row:
                #         Ellipse(pos=(self.grid_width*position - 5, self.height - (self.grid_height*row_num) - 5), size=(10, 10))
                #         if row_num == 1:
                #             self.add_widget(Label(center=(self.grid_width*position, self.height - (self.grid_height*row_num) + 20), text="%s" %('-' if not inputs else inputs[i])))
                #         elif row_num == self.num_rows:
                #             self.add_widget(Label(center=(self.grid_width*position, self.height - (self.grid_height*row_num) - 20), text="%s" %('-' if not outputs else outputs[i])))
                #         i+=1

            # i = 0
            # for position in self.grid_positions:
            #     Label(pos=(self.grid_width*self.grid_positions[0][i], self.height - (self.grid_height)), text=str(val))
            #     i += 1

            # # Draw Input Layer


            # layer = 0
            # for vec_layer in self.neural_network.vec_layers:
            #     layer += 1
            #     # print "layer %d" %layer
            #     layer_neuron = 0
            #     Label(pos=self.center, text="%s" %self.grid_positions)
            #     for neuron in vec_layer.vec_neurons:
            #         layer_neuron += 1
            #         # print "neuron %d" %layer_neuron
            #         # print neuron.vec_weight
            #         Ellipse(pos=(100*layer_neuron, 100*layer), size=(10, 10))


    def createNet(self, num_inputs=2, num_outputs=2, num_hidden_layers=2, neurons_per_hidden_layer=4):
        self.neural_network = CNeuralNet(num_inputs, num_outputs, num_hidden_layers, neurons_per_hidden_layer)
        self.num_rows = self.neural_network.num_hidden_layers + 2
        self.num_columns = max(self.neural_network.neurons_per_hidden_layer,
            self.neural_network.num_inputs,
            self.neural_network.num_outputs)
        # self.defineGridPositions()
        self.grid_height = self.height / (self.num_rows + 1)
        self.grid_width = self.width / (self.num_columns + 1)

        self.linkDisplayToNetwork()
        self.draw()
    pass

class NeuralNetApp(App):
    def build(self):
        network = NeuralDisplay(size=(Window.width, Window.height))
        network.createNet(num_inputs=64, num_outputs=1, num_hidden_layers=1, neurons_per_hidden_layer=24)
        # network.update([1,0])
        Clock.schedule_interval(network.update, 1.0)
        # network.draw()
        # return Label(text='Hello world')
        return network 


if __name__ == '__main__':
    NeuralNetApp().run()