#!/usr/bin/python
import kivy
kivy.require('1.8.0') # replace with your current kivy version !

from kivy.config import Config
Config.set('kivy', 'keyboard_mode', 'systemandmulti')
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
import time
from neuralnet import CNeuralNet



CONFIG = {
    'num_inputs': 7,
    'num_outputs': 10,
    'num_hidden_layers': 1,
    'neurons_per_hidden_layer': 16
}
BINARY_SIZE = '0000000'
DIGITS = ['1110111',
    '0010001',
    '0111110',
    '0111011',
    '1011001',
    '1101011',
    '1101111',
    '0110001',
    '1111111',
    '1111011']

class NeuronCell(Widget):
    neuron = ObjectProperty(None)
    value = NumericProperty(0)
    layer = StringProperty('hidden')
    r = NumericProperty(0.5)
    g = NumericProperty(0.5)
    b = NumericProperty(0.5)

    def load_widget(self, grid_x, grid_y):
        (self.grid_x, self.grid_y) = (grid_x, grid_y)
        (real_x, real_y) = (self.parent.grid_width*grid_x, self.parent.height - (self.parent.grid_height*grid_y))

        with self.canvas:
            self.cell = Ellipse(pos=(real_x-5, real_y-5), size=(10, 10))

        if self.layer == 'input':
            self.label = Label(center=(real_x, real_y + 20), text="-")
            self.add_widget(self.label)
        elif self.layer == 'output':
            self.label = Label(center=(real_x, real_y - 20), text="-")
            self.add_widget(self.label)

    def update(self, value=False):
        if value and self.label:
            self.value = float(value)
            self.label.text = "%.2f" %self.value

            if self.value > 0.95 and self.cell:
                self.r = 0.0
                self.g = 1.0
                self.b = 0.0
                # with self.canvas:
                #     Color(0,1,0)
                #     mypos = self.cell.pos
                #     self.cell = Ellipse(pos=mypos, size=(10, 10))
                #     Color(1,1,1)
            elif self.value < 0.05 and self.cell:
                self.r = 1.0
                self.g = 0.0
                self.b = 0.0
                # with self.canvas:
                #     Color(1,0,0)
                #     mypos = self.cell.pos
                #     self.cell = Ellipse(pos=mypos, size=(10, 10))
                #     Color(1,1,1)
            else:
                self.r = 0.5
                self.g = 0.5
                self.b = 0.5
                # with self.canvas:
                #     Color(0.5,0.5,0.5)
                #     mypos = self.cell.pos
                #     self.cell = Ellipse(pos=mypos, size=(10, 10))
                #     Color(1,1,1)
    pass    

class NeuralDisplay(Widget):

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

    def getLayerPositions(self, n, m):
        if n > m:
            return []

        diff = m-n
        start_x = 1 + (diff / 2)

        if diff % 2:
            start_x += 0.5

        return [ start_x + i for i in range(n) ]

    def update(self, inputs=False):
        outputs = self.neural_network.update(inputs)
        self.epoch_label.text = "Epoch: %s" %self.epoch

        i=0
        for neuron_wid in self.layers[0]:
            neuron_wid.update(str(inputs[i]))
            i+=1
        i=0
        for neuron_wid in self.layers[-1]:
            neuron_wid.update(str(outputs[i]))
            i+=1

    def loadNet(self, neural_network):
        self.neural_network = neural_network
        self.epoch = 1
        self.epoch_label = Label(text="Epoch: %s" %self.epoch, center=(self.width / 2, self.height))
        self.add_widget(self.epoch_label)
        self.num_rows = self.neural_network.num_hidden_layers + 2
        self.num_columns = max(self.neural_network.neurons_per_hidden_layer,
            self.neural_network.num_inputs,
            self.neural_network.num_outputs)
        # self.defineGridPositions()
        self.grid_height = self.height / (self.num_rows + 1)
        self.grid_width = self.width / (self.num_columns + 1)

        self.linkDisplayToNetwork()
    
    pass


class DigitalDisplay(Widget):
    node1 = ObjectProperty(0)
    node2 = ObjectProperty(0)
    node3 = ObjectProperty(0)
    node4 = ObjectProperty(0)
    node5 = ObjectProperty(0)
    node6 = ObjectProperty(0)
    node7 = ObjectProperty(0)

    def init(self):
        self.nodes = [
            self.node1,
            self.node2,
            self.node3,
            self.node4,
            self.node5,
            self.node6,
            self.node7,
        ]

    def update(self, inputs):
        if len(inputs) == len(self.nodes):
            for i in range(len(inputs)):
                self.nodes[i].value = inputs[i]

    pass

class DigitalDisplayNode(Widget):
    value = NumericProperty(0)
    pass

class DisplayController(Widget):  
    integer = NumericProperty(0)

    def __init__(self, **kwargs):
        super(DisplayController, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.look_at = 0

    def on_touch_down(self, touch):

        if self.look_at < 9:
            self.look_at += 1
        else:
            self.look_at = 0

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == '0':
            self.look_at = 0
        elif keycode[1] == '1':
            self.look_at = 1
        elif keycode[1] == '2':
            self.look_at = 2
        elif keycode[1] == '3':
            self.look_at = 3
        elif keycode[1] == '4':
            self.look_at = 4
        elif keycode[1] == '5':
            self.look_at = 5
        elif keycode[1] == '6':
            self.look_at = 6
        elif keycode[1] == '7':
            self.look_at = 7
        elif keycode[1] == '8':
            self.look_at = 8
        elif keycode[1] == '9':
            self.look_at = 9
        return True

    def update(self, arg):
    # for i in range(128):
        # binary_i = bin(self.integer)[2:]
        # padded_binary_i = BINARY_SIZE[:-len(binary_i)] + binary_i
        # self.integer += 1

        for i in range(1):
            for j in range(10):
                inputs = DIGITS[j]
                # inputs = [1,0]
                expected = [ 1 if j == k else 0 for k in range(10) ]
                inputs = [ int(bool) for bool in inputs ]
                outputs = self.network.neural_network.update(inputs)
                errors = self.network.neural_network.backpropErrors(outputs, expected)
                self.network.neural_network.adjustWeights(inputs, errors)

        self.network.epoch += 1
        digit = DIGITS[self.look_at]
        # digit = DIGITS[4]
        inputs = [ int(bool) for bool in digit]

        # inputs = [ randint(0,1) for x in range(self.network.neural_network.num_inputs) ]
        self.network.update(inputs)
        # self.add_widget(Label(text=str( ["%.2f" %w for w in self.network.neural_network.getWeights()] )))
        self.digit.update(inputs)


    pass


class NeuralNetApp(App):

    def on_pause(self):
      # Here you can save data if needed
        return True

    def on_resume(self):
      # Here you can check if any data needs replacing (usually nothing)
        pass
    def build(self):

        neural_network = CNeuralNet(
            CONFIG['num_inputs'],
            CONFIG['num_outputs'],
            CONFIG['num_hidden_layers'],
            CONFIG['neurons_per_hidden_layer']
        )

        control = DisplayController()
        control.digit = DigitalDisplay(pos = (0, Window.height-200), size=(200,200))
        control.digit.init()
        control.network = NeuralDisplay(size=(Window.width, Window.height-200))
        control.network.loadNet(neural_network=neural_network)

        control.add_widget(control.digit)
        control.add_widget(control.network)


        Clock.schedule_interval(control.update, 0.05)

        return control 


if __name__ == '__main__':
    NeuralNetApp().run()    
