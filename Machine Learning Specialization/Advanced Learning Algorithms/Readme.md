# Advanced Learning Algorithms

## Neuron Networks

a vector activation = g(x) = sigmoid =  1/(1+e<sup>-x</sup>)

a layer can have one or more neurons  
                                            a  
Input layer ---------> hidden layer/layers---------> output layer  
Feature Vector X                                   Vector Y  
Layer 0                 Layer 1/2/3                   Layer 4  

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

g = sigmoid function =  1/(1+e<sup>-z</sup>)  
z = W.x + b  
<pre>
for layer 1 vec a<sup>[1]</sup> = [0.3,0.7,0.2]  
a<sub>1</sub><sup>[1]</sup> = g(W<sub>1</sub><sup>[1]</sup>.x+b<sub>1</sub><sup>[1]</sup>)=0.3  
a<sub>2</sub><sup>[1]</sup> = g(W<sub>2</sub><sup>[1]</sup>.x+b<sub>2</sub><sup>[1]</sup>)=0.7  
a<sub>3</sub><sup>[1]</sup> = g(W<sub>3</sub><sup>[1]</sup>.x+b<sub>3</sub><sup>[1]</sup>)=0.2  

for layer 2 vec a<sup>[2]</sup> = [0.84]  
a<sub>1</sub><sup>[2]</sup> = g(W<sub>1</sub><sup>[2]</sup>.a<sup>[1]</sup>+b<sub>2</sub><sup>[2]</sup>)=0.84  


if (a<sup>[2]</sup> >= 0.5)  
yes: ŷ = 1  
no: ŷ =0  
</pre>
## Complex neural networks activation vector computation

a<sub>j</sub><sup>[l]</sup> = g(vec W<sub>j</sub><sup>[l]</sup>.vec a<sup>[l-1]</sup> + b<sub>j</sub><sup>[l]</sup>)

l layer  
j unit of neuron  
g activation function = sigmoid function  
vec X = vec a<sup>[0]</sup>  

## Neural network to make predections or inferences

### handwritten digit recog

8x8 pixel -
<pre>
255 255 255 255 255 255 255 255  
255 255 255     255 255 255 255  
255 255         255 255 255 255  
255 255 255     255 255 255 255  
255 255 255     255 255 255 255  
255 255 255     255 255 255 255  
255 255             255 255 255  
255 255 255 255 255 255 255 255  
</pre>
2 hidden layers

Layer 0 = 8x8 = 64 values in vec X  
Layer 1 = 25 units  
Layer 2 = 15 uints  
Layer 3 = 1 unit  
<pre>
vec a<sup>[1]</sup> = [ g(W<sub>1</sub><sup>[1]</sup>.vec a<sup>[0]</sup>+b<sub>1</sub><sup>[1]</sup>),  
                    g(W<sub>2</sub><sup>[1]</sup>.vec a<sup>[0]</sup>+b<sub>2</sub><sup>[1]</sup>)  
                    :  
                    :  
                     g(W<sub>25</sub><sup>[1]</sup>.vec a<sup>[0]</sup>+b<sub>25</sub><sup>[1]</sup>) ]  


vec a<sup>[2]</sup> = [ g(W<sub>1</sub><sup>[2]</sup>.vec a<sup>[1]</sup>+b<sub>1</sub><sup>[2]</sup>),  
                    g(W<sub>2</sub><sup>[2]</sup>.vec a<sup>[1]</sup>+b<sub>2</sub><sup>[2]</sup>)  
                    :  
                    :  
                     g(W<sub>15</sub><sup>[2]</sup>.vec a<sup>[1]</sup>+b<sub>25</sub><sup>[2]</sup>) ]  

a<sup>[3]</sup> = [ g(W<sub>1</sub><sup>[3]</sup>.vec a<sup>[2]</sup>+b<sub>1</sub><sup>[3]</sup>)]  

if (a<sup>[3]</sup> >= 0.5)  
yes: ŷ = 1  
no: ŷ =0  
</pre>
This is called forward propagation as we are moving form left to right

## Tensorflow

### For coffee roasting check if this will give good coffee or not

Temp = 200  
duration = 17

```python
x = np.array([[200.0,17.0]])
layer_1 = Dense(units=3, activation='sigmoid')  
a1 = layer_1(x)

layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)

if a2 >= 0.5:
  yhat = 1
else:
  yhat = 0
```

### For digit prediction

```python
x = np.array([[255,255,.......255]])
layer_1 = Dense(units=25, activation='sigmoid')
a1 = layer_1(x)

layer_2 = Dense(units=15, activation='sigmoid')
a2 = layer_2(a1)


layer_3 = Dense(units=1, activation='sigmoid')
a3 = layer_2(a2)

if a3 >= 0.5:
  yhat = 1
else:
  yhat = 0
```

## Data representation in numpy for tensor flow

2D matrix 2X3
<pre>
1 2 3  
4 5 6

x = np.array([[1,2,3],  
              [4,5,6]])

2D matrix 1X2

x = np.array([[200.0,17.0]])

2D matrix 2X1

x = np.array([[200.0],  
              [17.0]])

1D Vector

x = np.array([200.0,17.0])

Linear regression we use vector  
tensor flow we use matrix

to convert from tensor to numpy matrix

a3.numpy()
</pre>

## Build neural network in tensor flow

```python
layer_1 = Dense(units=3, activation='sigmoid')
layer_2 = Dense(units=1, activation='sigmoid')

model = Sequential([layer_1,layer_2])

x = np.array([[200.0, 17.0],
              [120.0,5.0],
              [425.0,20.0],
              [212.0,18.0]])

y = np.array([1,0,0,1])

model.compile(...)

//  Train data
model.fit(x,y)

//Predict
model.predict(x_new)
```

```python
layer_1 = Dense(units=25, activation='sigmoid')
layer_2 = Dense(units=15, activation='sigmoid')
layer_3 = Dense(units=1, activation='sigmoid')

model = Sequential([Dense(units=25, activation='sigmoid'),
                    Dense(units=15, activation='sigmoid'),
                     Dense(units=1, activation='sigmoid')])

x = np.array([[255,255,.......255],
              [255,255,.......255]])

y = np.array([1,0])

model.compile(...)

//  Train data
model.fit(x,y)

//Predict
model.predict(x_new)
```

## Forward propagation from scratch

<pre>

eg - a0 is vec X is layer 0
Layer 1 is 3 units
a1 =  [a1_1,a1_2,a1_3]

a2 = [a2_1]

```python 
x = np.array([200,17])
```

a<sub>1</sub><sup>[1]</sup> = g(W<sub>1</sub><sup>[1]</sup> . vec x + b<sub>1</sub><sup>[1]</sup>)

``` python
x = np.array([200,17])

w1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1,x)+b
a1_1 = sigmoid(z1_1)

```

a<sub>2</sub><sup>[1]</sup> = g(W<sub>2</sub><sup>[1]</sup> . vec x + b<sub>2</sub><sup>[1]</sup>)

``` python
x = np.array([200,17])

w1_2 = np.array([-3,4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2,x)+b
a1_2 = sigmoid(z1_2)

```

a<sub>3</sub><sup>[1]</sup> = g(W<sub>3</sub><sup>[1]</sup> . vec x + b<sub>3</sub><sup>[1]</sup>)

``` python
x = np.array([200,17])

w1_3 = np.array([5,-6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3,x)+b
a1_3 = sigmoid(z1_3)

a1 = np.array([a1_1,a1_2,a1_3])

```

a<sub>1</sub><sup>[2]</sup> = g(W<sub>1</sub><sup>[2]</sup> . vec x + b<sub>1</sub><sup>[2]</sup>)

```python
w2_1 = np.array([-7,8])
b2_1 = np.array([3])
z2_1 = np.dot(w2_1,a1) + b2_1
a2_1 = sigmoid(z2_1)
```
</pre>

## Forward propagation in NumPy

<pre>

W1 = [ W<sub>1</sub><sup>[1]</sup>,W<sub>2</sub><sup>[1]</sup>,W<sub>3</sub><sup>[1]</sup> ]

2x1 matrix
W<sub>1</sub><sup>[1]</sup> = [ 1 ]
                              [ 2 ]
2x1 matrix
W<sub>2</sub><sup>[1]</sup> = [ -3 ]
                              [ 4  ]
2x1 matrix
W<sub>3</sub><sup>[1]</sup> = [  5 ]
                              [ -6 ]

```python
W1 = np.array([
              [1,-3,5 ]
              [2,4,-6]])
```
b1 = [b<sub>1</sub><sup>[1]</sup>,b<sub>2</sub><sup>[1]</sup>,b<sub>3</sub><sup>[1]</sup>]

```python
b1 = np.array([-1,1,2])

a_in = np.array([-2,4])

def dense(a_in,W,b,g):
  units = W.shape[1]
  a_out = np.zeros(units)
  for j in range(units):
    w = W[:,j]
    z = np.dot(w,a_in) +b[j]
    a_out[j] = g(z)
  return a_out

def sequential(a_in):
  a1 = dense(a_in,W1,b1)
  a2 = dense(a1,W2,b2)
  a3 = dense(a2,W3,b3)
  a4 = dense(a3,W4,b4)
  f_x = a4
  return f_x
```
</pre>
