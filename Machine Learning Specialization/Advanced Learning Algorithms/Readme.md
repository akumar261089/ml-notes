# Advanced Learning Algorithms

## Neuron Networks

a vector activation = f(x) = sigmoid

a layer can have one or more neurons
                                            a
Input layer ---------> hidden layer/layers---------> output layer
Feature Vector X                                   Vector Y

final layer or output layer have one neuron

Number of hidden layers and number of neurons in hidden layers are architecture of neural network

### Demand prediction

X = [price, shipping cost, marketing, material]
L1 activations  =[affordablity(price, shipping cost),awareness(marketing), quality(Price, material)]
OL = probablity of being top seller

### Face recognition

eg 1000x1000 pixel image
vector X = 1000000 elements

L1 = looking for short lines or edges
L2 = group together to figure part of faces - eyes nose
L3 = different part of faces - shapes and size

feeding different data NL learns automatically

### how to build a layer of neuron

for layer 1 vec a<sup>[1]</sup> = [0.3,0.7,0.2]
a<sub>1</sub><sup>[1]</sup> = g(W<sub>1</sub><sup>[1]</sup>.x+b<sub>1</sub><sup>[1]</sup>)=0.3
a<sub>2</sub><sup>[1]</sup> = g(W<sub>2</sub><sup>[1]</sup>.x+b<sub>2</sub><sup>[1]</sup>)=0.7
a<sub>3</sub><sup>[1]</sup> = g(W<sub>3</sub><sup>[1]</sup>.x+b<sub>3</sub><sup>[1]</sup>)=0.2

for layer 2 vec a<sup>[2]</sup> = [0.84]
a<sub>1</sub><sup>[2]</sup> = g(W<sub>1</sub><sup>[2]</sup>.a<sup>[1]</sup>+b<sub>2</sub><sup>[2]</sup>)=0.84

if (a<sup>[2]</sup> >= 0.5)
yes: ŷ = 1
no: ŷ =0