# Machine Learning Specialization

## Machine learning

<https://www.coursera.org/specializations/machine-learning-introduction>

> Field of study that gives computers the ability to learn without being explicitly programmed. -- Arthur Samuel.

### Types of Machine learning

- Supervised Learning
- Unsupervised Learning
- Recommender Systems
- Reinforcement Learning

### Supervised Machine learning

Supervised machine learning or more commonly, supervised learning, refers to algorithms that learn x to y or input to output mappings.
eg - Spam Filter, Speech recognition, machine translation(language), online advertising, self driving cars, visual inspection

### Type of Supervised learning based on output

- Regression(can predict any number(infinite) as output) eg - housing prices
- Classification(can predict only limited set of possible outputs like -0,1,2 or category/class two or more) eg - Type of cancer

### Unsupervised learning

Were given data that isn't associated with any output labels y and find something interseting in unlableled data eg - Google news

### Type of Unsupervised learning

- Clustering(Group similar data points together) eg - Google news, DNA microarray clustering, customer grouping
- Anomaly Detection(Find unusual data points) eg - Fraud detection
- Dimensionality Reduction(Compress data using fewer numbers)

### Linear regression Type of regression model

We have a training set with x as feature or input variable & we have y as output variable or target variable. m is number of training examples
Single training example is called as (x,y) and i<sup>th</sup> example is called a (x<sup>(i)</sup>,y<sup>(i)</sup>)

#### Linear regression components

1. Training set
    - Features
    - Targets
2. Learning Algorithm
    - Function f
    x->f-> ŷ

Function as a straight line- leaner regression with one variable(univariate) f<sub>w,b</sub>(X)=wX+b= ŷ
where w&b are parameters

#### Cost Function

How well the model is doing. and tune parameters.
Cost Func : Squared error cost function
    J(w,b) =  1/(2m) * sum over i = 1 to m ( ŷ<sup>(i)</sup> - y<sup>(i)</sup>)<sup>2</sup>
    J(w,b) =  1/(2m) * sum over i = 1 to m ( f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>

#### Minimize cost function - Training Linear regression model Gradient decent

min J(w,b), α is learning rate
w<sub>new</sub>=w-α * dJ(w,b)/dw
b<sub>new</sub>=b-α * dJ(w,b)/db
both update takes place at same time, similarly if we have multiple parameters they will be updated at same time or in one step

batch gradient decent = looking at all training data

#### Multiple features linear regression

f<sub>W,b</sub>(X)=w<sub>1</sub>X<sub>1</sub>+w<sub>2</sub>X<sub>2</sub>+w<sub>3</sub>X<sub>3</sub>+w<sub>4</sub>X<sub>4</sub>+b= ŷ
w = np.array([w<sub>1</sub>+w<sub>2</sub>+w<sub>3</sub>+w<sub>4</sub>])
x = np.array([X<sub>1</sub>+X<sub>2</sub>+X<sub>3</sub>+X<sub>4</sub>])
f<sub>W,b</sub>(X)=np.dot(w,x) + b

#### Feature scaling

valuse should be between -1 < x < 1 by using ans dhould not be very small

- divide by max
- mean normalization
- z score normalization

#### feature engineering

eg - instead of length & width we can directly use area. l , w, a we can get better model by combining features

#### check gradient decent is converging

do we need more steps or not

- learing curve
- Automatic convergence test

Learing rate  -- 0.001 or 0.003 or 0.01 or 0.03 or 0.1 or 0.3 or 1 

#### Classification

logistic regression - sigmoid function

Decision boundary

Cost function for logisdtic regression -squared error cost
J(w,b) =  (1/m) * sum over i = 1 to m 1/2( ŷ<sup>(i)</sup> - y<sup>(i)</sup>)<sup>2</sup>

Loss funct

- log(f<sub>W,b</sub>(X<sup>(i)</sup>)) if y<sup>(i)</sup> =1

- log(1 - f<sub>W,b</sub>(X<sup>(i)</sup>)) if y<sup>(i)</sup> =0

Gradient decent on new cost function

#### Over fitting problem

- collect more training data
- fewer features - feature selection
- Regularization - gentally reduce impact of parameters- small values of w by adding sum of all w in cost function resultinh reducing value of w in gradient decent