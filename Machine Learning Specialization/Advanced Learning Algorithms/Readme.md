# [Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms)

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
L1 activations  =[affordability(price, shipping cost),awareness(marketing), quality(Price, material)]  
OL = probability of being top seller

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

## Neural network to make predictions or inferences

### handwritten digit recognition

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
Layer 2 = 15 units  
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
</pre>

```python
x = np.array([200,17])
```

<pre>
a<sub>1</sub><sup>[1]</sup> = g(W<sub>1</sub><sup>[1]</sup> . vec x + b<sub>1</sub><sup>[1]</sup>)
</pre>

``` python
x = np.array([200,17])

w1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1,x)+b
a1_1 = sigmoid(z1_1)

```

<pre>
a<sub>2</sub><sup>[1]</sup> = g(W<sub>2</sub><sup>[1]</sup> . vec x + b<sub>2</sub><sup>[1]</sup>)

</pre>

``` python
x = np.array([200,17])

w1_2 = np.array([-3,4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2,x)+b
a1_2 = sigmoid(z1_2)

```

<pre>
a<sub>3</sub><sup>[1]</sup> = g(W<sub>3</sub><sup>[1]</sup> . vec x + b<sub>3</sub><sup>[1]</sup>)
</pre>

``` python
x = np.array([200,17])

w1_3 = np.array([5,-6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3,x)+b
a1_3 = sigmoid(z1_3)

a1 = np.array([a1_1,a1_2,a1_3])

```

<pre>
a<sub>1</sub><sup>[2]</sup> = g(W<sub>1</sub><sup>[2]</sup> . vec x + b<sub>1</sub><sup>[2]</sup>)
</pre>

```python
w2_1 = np.array([-7,8])
b2_1 = np.array([3])
z2_1 = np.dot(w2_1,a1) + b2_1
a2_1 = sigmoid(z2_1)
```

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
</pre>

```python
W1 = np.array([
              [1,-3,5 ]
              [2,4,-6]])
```

<pre>
b1 = [b<sub>1</sub><sup>[1]</sup>,b<sub>2</sub><sup>[1]</sup>,b<sub>3</sub><sup>[1]</sup>]
</pre>

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

[1,0,1]

## Vectorized implementation of forward propagation

```python
X= np.array([[200,17]])
W=np.array([[1,-3,5],
            [-2,4,-6]])
B = np.array([[-1,1,2]])

def dense (A_in,W,B):
    Z= np.matmul(A_in,W) + B
    A_out =g(Z)
    return A_out
```

[[1,0,1]]

## Matrix multiplication in  numpy

```python

A=np.array([1,-1,0.1],
           [2,-2,0.2])
AT = np.array([1,2],
              [-1,-2],
              [0.1,0.2])

//AT= A.T

W= np.array([3,5,7,9],
            [4,6,8,0])
Z = np.matmul(AT,W)
```

## AGI - artificial general intelligence

1. ANI - artificial narrow intelligence eg - smart speaker, self driving car, we search
2. AGI - artificial general intelligence - Do anything a human can do

## Tensorflow implementation

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow..keras.losses import BinaryCrossentropy

// Step one specify the model how to compute
model = Sequential([
  Dense(units=25,activation='sigmoid')
  Dense(units=15,activation='sigmoid')
  Dense(units=1,activation='sigmoid')
])

//Step two compile the model with loss function
model.compile(loss=BinaryCrossentropy())

//Step three fit/train the model with training data  
model.fit(X,Y,epoch=100)
//epochs : number of steps in gradient decent
```

## Model training steps

<pre>

1. Specify how to compute output given input X and parameters w,b
    f<sub>w,b</sub>(X) =?

    Logistic Regression
    z = np.dot(w,x) +b
    f_x = 1/(1+np.exp(-z))

    Neural Network
    model = Sequential([
      Dense(...)
      Dense(...)
    ])

2. Specify loss function and cost
   L(f<sub>w,b</sub>(X),y)
   J(W,b) = 1/m(Sum for i 0 to m L(f<sub>w,b</sub>(X<sup>(i)</sup>),y<sup>(i)</sup>))

    Logistic Regression
    loss = -y*np.log(f_x) - (1-y)*np.log(1-f_x)

    Neural Network
    model.compile(loss=BinaryCrossentropy())

    for regression we can use mean square error loss
    from tensorflow..keras.losses import MeanSquareError
    model.compile(loss=MeanSquareError())

3. Train on data to minimize J(W,b)

    Logistic Regression
    w = w - alpha*dj_dw
    b = b - alpha*dj_db

    Neural Network
    // compute derivatives for gradient decent using back propagation
    model.fit(X,Y,epoch=100)

</pre>

## Activation Functions

We will choose activation function based on our requirement

- Output layer :

<pre>
Binary Classification can use Sigmoid y can be (0,1)
Regression can use linear y can be (- or +)
Regression can use ReLU where y can be (0 or +)
</pre>

- Hidden Layers :

<pre>
Mostly used ReLU as it is not flat
Sigmoid used sometimes
</pre>

don't use linear activation functions in hidden layers

## Multi class classification problem

Where we can have multiple classes eg - hand written number identification - number can be 0,1,2 etc or check defects
so any scenario where more than one possible outputs but small finite number

For that we can have n number of outputs

### Softmax regression

Logistic regression can have only 2 possible outputs
as function used is

```python
a1=g(z) = 1/(1+np.exp(-z))
```

in case of softmax we can have n number of outputs example for 4  

a1 = g(z1) = e<sup>z1</sup>/(e<sup>z1</sup> + e<sup>z2</sup> + e<sup>z3</sup> + e<sup>z4</sup> )  
a2 = g(z2) = e<sup>z2</sup>/(e<sup>z1</sup> + e<sup>z2</sup> + e<sup>z3</sup> + e<sup>z4</sup> )  
a3 = g(z3) = e<sup>z3</sup>/(e<sup>z1</sup> + e<sup>z2</sup> + e<sup>z3</sup> + e<sup>z4</sup> )  
a4 = g(z4) = e<sup>z4</sup>/(e<sup>z1</sup> + e<sup>z2</sup> + e<sup>z3</sup> + e<sup>z4</sup> )  

Generalize  
a<sub>j</sub> = e<sup>Zj</sup>/sum from k= 1 to N e<sup>Zk</sup>

#### Cost calculation

loss for logistic regression = -y log a<sub>1</sub> - (1-y)log(1- a<sub>1</sub>)
J(W,b) = average loss

loss(a<sub1>1</sub>,...a<sub1>N</sub>,y) = -log a<sub>N</sub> if y = N

### Neural network for multi class

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow..keras.losses import SparseCategoricalCrossentropy

// Step one specify the model how to compute
model = Sequential([
  Dense(units=25,activation='relu')
  Dense(units=15,activation='relu')
  Dense(units=10,activation='softmax')
])

//Step two compile the model with loss function
model.compile(loss=SparseCategoricalCrossentropy())

//Step three fit/train the model with training data  
model.fit(X,Y,epoch=100)
//epochs : number of steps in gradient decent
```

but we can improve this by using different method and avoid numerical round off error caused while computing in machines

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow..keras.losses import SparseCategoricalCrossentropy

// Step one specify the model how to compute
model = Sequential([
  Dense(units=25,activation='relu')
  Dense(units=15,activation='relu')
  Dense(units=10,activation='linear')
])

//Step two compile the model with loss function
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

//Step three fit/train the model with training data  
model.fit(X,Y,epoch=100)
//epochs : number of steps in gradient decent

logits = model(X)

f_x = tf.nn.softmax(logits)
```

Improved method for logistic regression

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow..keras.losses import BinaryCrossentropy

// Step one specify the model how to compute
model = Sequential([
  Dense(units=25,activation='relu')
  Dense(units=15,activation='relu')
  Dense(units=1,activation='linear')
])

//Step two compile the model with loss function
model.compile(loss=BinaryCrossentropy(from_logits=True))

//Step three fit/train the model with training data  
model.fit(X,Y,epoch=100)
//epochs : number of steps in gradient decent

logits = model(X)

f_x = tf.nn.sigmoid(logits)

```

## Multi label classification problem

if in once input we have multiple labels eg - scanning a photo with multiple objects like human,car,bus etc

We can combine these using sigmoid activation for output layer and result is a vector

## Advanced Model optimization algos

Adam this is safe choice to use

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
  loss=tf.keras.losses.SparseCategoryCrossentropy(from_logits=True))
```

## Tips to build machine learning systems

### 1st example regular linear regression on housing prices

J(W,b) = 1/2m(sum for i = 0 to m (f<sub>W,b</sub>(X<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup> - lambda/2m(sum for j =1 to n W<sub>j</sub><sup>2</sup>) )

#### but in this we get large errors what to do next

- Get mode training examples
- Try smaller sets of features
- Try getting additional features
- try adding polynomial features(x<sub>1</sub><sup>2</sup>,x<sub>2</sub><sup>2</sup>,x<sub>1</sub>,x<sub>2</sub> etc)
- Try decreasing lambda
- try increasing lambda

#### Evaluate machine learning model

Split data into training set and test set(70-30 or 80-20)

##### Train test procedure for regression

###### Fit parameters by minimizing cost function J(W,b)  

J(W,b) = min<sub>W,b</sub>(1/2m<sub>train</sub>(sum where i =1 to m (f<sub>W,b</sub>(X<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup> - lambda/2m(sum for j =1 to n W<sub>j</sub><sup>2</sup>)))

###### Compute test error

J<sub>test</sub>(W,b) = 1/2m<sub>test</sub>(sum where i =1 to m<sub>test</sub> (f<sub>W,b</sub>(X<sub>test</sub><sup>(i)</sup>)-y<sub>test</sub><sup>(i)</sup>)<sup>2</sup>)

###### Compute train error

J<sub>train</sub>(W,b) = 1/2m<sub>train</sub>(sum where i =1 to m<sub>train</sub> (f<sub>W,b</sub>(X<sub>train</sub><sup>(i)</sup>)-y<sub>train</sub><sup>(i)</sup>)<sup>2</sup>)

##### Train test procedure for classification

###### Fit parameters by minimizing cost function J(W,b)  to find w,b

J(W,b) = -1/m(sum where i =1 to m (y<sup>(i)</sup>log(f<sub>W,b</sub>(X<sup>(i)</sup>)))+(1-y<sup>(i)</sup>)log( 1 - f<sub>W,b</sub>(X<sup>(i)</sup>)) + lambda/2m(sum for j =1 to n W<sub>j</sub><sup>2</sup>)))

###### Compute test error for classification

J<sub>test</sub>(W,b) = 1/m<sub>test</sub>(sum where i =1 to m<sub>test</sub> (y<sub>test</sub><sup>(i)</sup>log(f<sub>W,b</sub>(X<sub>test</sub><sup>(i)</sup>)))+(1-y<sub>test</sub><sup>(i)</sup>)log( 1 - f<sub>W,b</sub>(X<sub>test</sub><sup>(i)</sup>)

###### compute train error for classification

J<sub>train</sub>(W,b) = 1/m<sub>train</sub>(sum where i =1 to m<sub>train</sub> (y<sub>train</sub><sup>(i)</sup>log(f<sub>W,b</sub>(X<sub>train</sub><sup>(i)</sup>)))+(1-y<sub>train</sub><sup>(i)</sup>)log( 1 - f<sub>W,b</sub>(X<sub>train</sub><sup>(i)</sup>)

#### Model selection

To Choose model we can check value of the cost function (J<sub>test</sub>(W,b)) for different degree. eg -

1. For d=1 f<sub>W,b</sub>(X) = W<sub>1</sub>X<sub>1</sub> + b
2. For d=2 f<sub>W,b</sub>(X) = W<sub>1</sub>X + W<sub>2</sub>X<sup>2</sup> + b
3. For d=3 f<sub>W,b</sub>(X) = W<sub>1</sub>X + W<sub>2</sub>X<sup>2</sup> + W<sub>3</sub>X<sup>3</sup> + b
.
.
.
10. For d=10 f<sub>W,b</sub>(X) = W<sub>1</sub>X + W<sub>2</sub>X<sup>2</sup> + ....+  W<sub>10</sub>X<sup>10</sup> + b

##### We should split data in 3 sets - training(60) cross validation(20) and test set (20)

In this case we can use (J<sub>cv</sub>(W,b)) to et cross validation error and thi will give more relevant output  
We can use this same model to choose neural network architecture

#### Running diagnostics in machine learning model - Bias & variance

Under fit = high Bias: J<sub>train</sub> is high and J<sub>cv</sub> is high  
over fit - high variance: J<sub>train</sub> is low and J<sub>cv</sub> is high, J<sub>cv</sub>>> J<sub>train</sub>  
high bias & high variance if J<sub>train</sub> is hgh and J<sub>cv</sub> >> J<sub>train</sub>
eg -
<pre>

d = 1,J_train is high, j_CV is high
d = 2,J_train is low, j_CV is low
d = 4,J_train is low, j_CV is high  

so we can see d =2 is a better solution
</pre>

#### Regularization and bias/variance

J(W,b) = 1/2m(sum for i = 0 to m (f<sub>W,b</sub>(X<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup> - lambda/2m(sum for j =1 to n W<sub>j</sub><sup>2</sup>) )  

if lambda is large curve will under fit as a result high bias  
if lambda is small curve will over fit as a result high variance  
so we take intermediate lambda it will be just right

to find correct value of lambda cross validation is helpful

1. Try lambda = 0
2. Try lambda = 0.01
3. Try lambda = 0.02
4. Try lambda = 0.04
5. Try lambda = 0.08

and check J<sub>cv</sub>(W<sup><1></sup>,b<sup><1></sup>)  
<pre>
lambda = 10,  J_train is high, j_CV is high
lambda = 0.08,J_train is low, j_CV is low
lambda = 0,   J_train is low, j_CV is high  
</pre>

#### Establishing a baseline level of performance
 