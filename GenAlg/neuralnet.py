#!/usr/bin/python

from random import uniform
import math

BINARY_SIZE = '0000000'
DIGITS = ['1110111',
    '1000100',
    '0111110',
    '0111011',
    '1011001',
    '1101011',
    '1101111',
    '0110001',
    '1111111',
    '1111011']

CONFIG = {
    'num_inputs': 7,
    'num_outputs': 10,
    'num_hidden_layers': 2,
    'neurons_per_hidden_layer': 6
}


### Neural Network with Back Propagation and Genetic Algorithm learning methods
# 
# NN and GenAlg stuff Ported from AI-JUNKIE's minesweeper tutorials
# Backprop drawn from various sources
#
###

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

    def getWeights(self):
        print "getWeights"
        weights = []
        for layer in self.vec_layers:
            for neuron in layer.vec_neurons:
                for weight in neuron.vec_weight:
                    weights.append(weight)        
        return weights


    def getNumberOfWeights():
        return (self.num_inputs * self.neurons_per_hidden_layer) + (self.num_hidden_layers)
        print "getNumberOfWeights"

    def putWeights(self, weights):
        print "putWeights"

        

    def update(self, inputs):
        # print "updating..."
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
                # The combined activation is first filtered through the sigmoid function
                # outputs.push_back(Sigmoid(netinput, CParams::dActivationResponse));
                outputs.append(self.sigmoid(netinput, 1))
                c_weight = 0
        return outputs

    def adjustWeights(self, inputs, errors):
        # print "adjusting..."
        # stores the resultant outputs from each layer
        # vector<double> outputs;
        # outputs = []
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
                # The combined activation is first filtered through the sigmoid function
                # outputs.push_back(Sigmoid(netinput, CParams::dActivationResponse));
                outputs.append(self.sigmoid(netinput, 1))
                c_weight = 0
            # print outputs
            N = 1
            for j in range(0, self.vec_layers[i].num_neurons):
                for k in range(len(self.vec_layers[i].vec_neurons[j].vec_weight)):
                    weight = self.vec_layers[i].vec_neurons[j].vec_weight[k]
                    # print "layer %s, neuron %s, weight %s" %(i,j,k)
                    # print "output %s, error %s, weight %s, input %s" %(outputs[j], errors[i][j], weight, (-1 if k == len(inputs) else inputs[k]))
                    # print "modify by %s" %(N * errors[i][j] * (-1 if k == len(inputs) else inputs[k]))
                    modification = N
                    # print errors[i]
                    modification *= errors[i][j]
                    modification *= (-1 if k == len(inputs) else inputs[k])
                    self.vec_layers[i].vec_neurons[j].vec_weight[k] = self.vec_layers[i].vec_neurons[j].vec_weight[k] + modification
                    # print (N*errors[i][k]*outputs[j])
                # self.vec_layers[i].vec_neurons[j].vec_weight = [ weight + (N * errors[i][j] * outputs[j]) for\
                #     weight in   self.vec_layers[i].vec_neurons[j].vec_weight ]
        return outputs

    def backpropErrors(self, outputs, expected):
        # Calculate output error(s)
        # print "backpropping..."
        error_layers = []
        if len(outputs) == len(expected):
            errors = []
            for output, target in zip(outputs, expected):
                error = output*(1-output)*(target-output)
                # print output, target, error
                errors.append(output*(1-output)*(target-output))
                # errors.append(target-output)
            error_layers.append(errors)

            # Back Propagate the errors in the hidden layers 
            for current_layer in reversed(self.vec_layers):
                errors = [ 0 for x in range(current_layer.vec_neurons[0].num_inputs + 1)]
                i=0
                for neuron in current_layer.vec_neurons:
                    self.error = error_layers[-1][i]
                    i+=1
                    # print zip(neuron.vec_weight, errors)
                    # for weight, error in zip(neuron.vec_weight, errors):
                    for j in range(len(errors)):
                        errors[j] += neuron.vec_weight[j]*self.error
                error_layers.append(errors)
        error_layers = error_layers[:-1]
        return error_layers[::-1]
        # Forwardly modify the weights using the errors

    def sigmoid(self, activation, response):
        # print "sigmoid"
        if activation < -45:
            netoutput = 0
        elif activation > 45:
            netoutput = 1
        else:
            netoutput = 1 / (1+math.exp(-activation))

        return netoutput


class CGenAlg():

    def __init__(self, pop_size, chromo_length,
                    mutation_rate = 0.2, crossover_rate = 0.7):
        # this holds the entire population of chromosomes
        # vector <SGenome> m_vecPop (i.e list of genomes)
        self.vec_pop = [ SGenome(vec_weight=[uniform(-1,1) for x in range(chromo_length)]) for n in range(pop_size) ]
        # for genome in self.vec_pop:
        #     genome.vec_weight = [ uniform(-1,1) for x in range(chromo_length) ]
        # size of population
        self.pop_size = pop_size
        # amount of weights per chromo
        self.chromo_length = chromo_length

        # fitness of population
        self.total_fitness = 0
        self.best_fitness = 9999999
        self.average_fitness = 0 
        self.worst_fitness = 0
        # generation counter
        sef.generation = 0

        # keeps track of the best genome
        self.fittest_genome = 0

        # probability that a chromosomes bits will mutate.
        # Try figures around 0.05 to 0.3 ish
        self.mutation_rate = mutation_rate

        # probability of chromosomes crossing over bits
        # 0.7 is pretty good
        self.crossover_rate = crossover_rate



    # //---------------------------------Mutate--------------------------------
    # //
    # //  mutates a chromosome by perturbing its weights by an amount not 
    # //  greater than CParams::dMaxPerturbation (here CONFIG.max_perturbation)
    # //-----------------------------------------------------------------------
    def mutate(self, chromo):
        # Like this returns a new list, how do I get the function to change chromo in place? i.e chromo.mutate()
        return [ gene + uniform(-1,1) * CONFIG['max_perturbation'] if uniform(0,1) < self.mutation_rate else gene for gene in chromo ]

    # //----------------------------------GetChromoRoulette()------------------
    # //
    # //  returns a chromo based on roulette wheel sampling
    # //
    # //-----------------------------------------------------------------------
    # SGenome GetChromoRoulette();
    def getChromoRoulette(self):
        fit_slice = uniform(0,1) * self.total_fitness

        fitness_so_far = 0
        for chromo in self.vec_pop:
            fitness_so_far += chromo.fitness
            if fitness_so_far >= fit_slice:
                return chromo


    # //-------------------------------------Crossover()-----------------------
    # //  
    # //  given parents and storage for the offspring this method performs
    # //  crossover according to the GAs crossover rate
    # //-----------------------------------------------------------------------
    def crossover(self, mum, dad):
        if uniform(0,1) > self.crossover_rate or mum == dad:
            return (mum, dad)

        crossover_point  = randint(0, self.chromo_length-1)

        baby1 = [ mum[i] if i < cp else dad[i] for i in range(self.chromo_length) ]
        baby2 = [ dad[i] if i < cp else mum[i] for i in range(self.chromo_length) ]

        return (baby1, baby2)

    # //-----------------------------------Epoch()-----------------------------
    # //
    # //  takes a population of chromosones and runs the algorithm through one
    # //   cycle.
    # //  Returns a new population of chromosones.
    # //
    # //-----------------------------------------------------------------------
    # vector<SGenome> Epoch(vector<SGenome> &old_pop);
    def epoch(self):

        new_pop = [ None for i in range(self.pop_size)]
        num_survivors = CONFIG['num_elite'] * CONFIG['num_copies_elite']
        if num_survivors > self.pop_size:
            return False

        new_pop[:num_survivors] = GrabNBest(CONFIG['num_elite'], CONFIG['num_copies_elite'])
        new_pop[num_survivors:] = [ SGenome(vec_weight = [ uniform(-1,1) for x in range(chromo_length) ]) for n in range(pop_size-num_survivors) ]


    # //-------------------------GrabNBest----------------------------------
    # //
    # //  This works like an advanced form of elitism by inserting NumCopies
    # //  copies of the NBest most fittest genomes into a population vector
    # //  Assumes the population is sorted by fitness
    # //
    # //--------------------------------------------------------------------
    def grabNBest(self, n_best, num_copies):
        elites = []
        for genome in reversed(self.vec_pop[-n_best:]):
            for n in range(num_copies):
                elites.append(SGenome(vec_weights=genome.vec_weights))
        return elites


    # //-----------------------CalculateBestWorstAvTot----------------------- 
    # //
    # //  calculates the fittest and weakest genome and the average/total 
    # //  fitness scores
    # //---------------------------------------------------------------------
    # void CalculateBestWorstAvTot();
    def calculateBestWorstAvTot(self):
        total_fitness = 0
        highest_so_far = 0
        lowest_so_far = 9999999




    void Reset();

    CGenAlg(int    popsize,
          double MutRat,
          double CrossRat,
          int    numweights);

    # -------------------accessor methods
    vector<SGenome> GetChromos()const{return m_vecPop;}

    double AverageFitness()const{return m_dTotalFitness / m_iPopSize;}

    double BestFitness()const{return m_dBestFitness;}





if __name__ == '__main__':
    # myint = 
    nn = CNeuralNet(7,10,2,16)
    print nn.vec_layers

    # binary_i = bin(expected)[2:]
    # padded_binary_i = BINARY_SIZE[:-len(binary_i)] + binary_i
    # expected += 1

    inputs = DIGITS[1]
    # inputs = [1,0]
    expected = [1]
    inputs = [ int(bool) for bool in inputs ]

    # inputs = [ randint(0,1) for x in range(self.network.neural_network.num_inputs) ]
    print inputs



    for i in range(1000):
        for j in range(10):
            inputs = DIGITS[j]
            # inputs = [1,0]
            expected = [ 1 if j == k else 0 for k in range(10) ]
            inputs = [ int(bool) for bool in inputs ]
            outputs = nn.update(inputs)
            errors = nn.backpropErrors(outputs, expected)
            nn.adjustWeights(inputs, errors)
            # print "Epoch: %d, Expected %s, Errors %s, Output %s" %(i, expected, errors[-1], outputs)
            if i % 50 == 0:
                print "Epoch: %d, Expected %s, Output %s" %(i, expected, ["%.2f" %output for output in outputs])
        if i % 50 == 0:
            print "\n"

    # for i in range(100):
    #     outputs = nn.update(inputs)
    #     # print "output: %s" %outputs
    #     # print "expected: %s" %expected
    #     errors = nn.backpropErrors(outputs, expected)
    #     print "Epoch: %d, Expected %s, Errors %s, Output %s" %(i, expected, errors[-1], outputs)
    #     # for layer in errors:
    #     #     print "errors: %s" %layer
    #     nn.adjustWeights(inputs, errors)

    # print outputs
    # print nn.vec_layers

    # layer = 0
    # for vec_layer in nn.vec_layers:
    #     layer += 1
    #     print "layer %d" %layer
    #     layer_neuron = 0
    #     for neuron in vec_layer.vec_neurons:
    #         layer_neuron += 1
    #         print "neuron %d" %layer_neuron
    #         print neuron.vec_weight

    # print nn.update([1,1])
