# Key points

## Loss Functions

### 1. Binary Cross function

#### What is Binary cross function

#### When Binary cross function is used

It is used for logistic regression

```python
from tensorflow..keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy())
```

### 2. Mean Square Error

#### What is Mean Square Error

#### When Mean Square Error is used

It is used for linear regression

```python
from tensorflow..keras.losses import MeanSquareError
model.compile(loss=MeanSquareError())
```

### 3. Sparse Categorical Cross Entropy

#### What is Sparse Categorical Cross Entropy

#### When Sparse Categorical Cross Entropy is used

```python
from tensorflow..keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy())
```

It is used with multi class problems

## Activation functions

### 1. Sigmoid function

#### What is sigmoid function

```python
g(z) = 1/(1+np.exp(-z))
```

#### When is sigmoid function used

### 2. ReLU function

#### What is ReLU function

Rectified Linear Unit

```python
g(z)= max(0,z)
```

#### When is ReLU function used

### 3. Linear function

#### What is Linear function

```python
a=g(z)=z=w.x+b
```

#### When is Linear function used

### 4. Softmax function

#### What is Softmax function

#### When is Softmax function used

### 5.  tan h activation function

### 6. LeakyReLU

### 7. Swish activation function

## Model optimization algos

1. Gradient decent

2. Adam Algorithm - adaptive movement estimation uses multiple learning rates - uses 11 learning rates automatically

## Types of hidden layers

1. Dense layers(all are inputs are shared by all units)

2. Convolutional Layers(few inputs are shared with few units)
 this is faster in computation as all inputs are not provided and needs less training as specific type of input is fet this also reduces cases of over fitting- network with convolution layers we call CNN
