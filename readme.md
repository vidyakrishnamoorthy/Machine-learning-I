# Machine learning algorithms
- Supervised learning
- Unsupervised learning
## Supervised learning
- give system mutliple right answers 
  based on multiple right answers system will detect the next rigth answer for an unanswered question
- Spam filtering
- Speech recognition
- Machine translation
- Online advertising
- Self-driving car
- Visual inspection

### Linear Regression
- Predict a number from infinitely many possibilities
- Can have infiite number of outputs for an input
- training set and testing set
- x as input (features) and y as output (target)
- m -> number of training examples (x,y); (x^i,y^i)
- n -> number of testing examples
- x -> f(x) -> Y; estimated -> Y-hat
- f(w,b)(x) = wx+b; f(x) = wx+b; b-> y intercept, x -> slope
 
#### Cost function
  - Parameters, weights, coefficients
  - Cost function
    - ![img.png](img.png)
  - Squared error cost function: Cost function divided by 2m to average the cost during to large tarining set
    - ![img_1.png](img_1.png)
  - Goal is to minimize the cost function J(w,b)
  - 

#### Gradient descent
- identifying the way to find the lowest possible cost
- identify the direction to decrease the cost from a given point iteratively
- Batch gradient descent
  - uses all training examples, not just a sample
  - 
##### Implementation
- w<sub>(i)</sub> = w<sub>(i-1)</sub> - a (d/dw) J(w<sub>(i-1)</sub>,b<sub>(i-1)</sub>)
- b<sub>(i)</sub> = b<sub>(i-1)</sub> - a (d/db) J(w<sub>(i-1)</sub>,b<sub>(i-1)</sub>)
- a -> learning rate; alpha
  - greater the a bigger the steps taken to find the lowest cost
  - always a positive number
  - if you are already at a local minimum, but the cost fn has 2 local minimums,
    - there is a chance to find 2 local minimums
    - Squared error cost funtion will only have 1 local minimum
- (d/dw) J(w(i-1),b(i-1)) -> derivative of the cost fn
  - decides direction to take the step and the size along with the a
  - slope of the function at a given point (y/x)
  - could be negative/ posyive slope
  - if negative slope, w increases, if positive. w decreases (moving it closer to the minimum)
- iterate until value reachs local minimum (converges)
    - value does not change much as compared to the previous value
- simultaneously update w and b

### Mutiple Linear Regression
- Using multiple features to find the target
- x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>...<sub>x<sub>j</sub> -> jth feature
- n -> number of features
- vector x(i) -> features of ith training example
- f<sub>(w,b)</sub>(x) = w1x1 + w2x2 + w3x3 + w4x4 + b
- vector x(i) -> features of ith training example
- x<sub>j</sub>(i) -> value of feature j in ith training example

#### Vectorization
```python
# initialization
w = np.array([1.0,2.0,3.0])
b=4
x=np.array([10,20,30])
```

```python
# Without vectorization
  f= 0
  for j in range(0,n):
    f=f+w[j] * x[j]
  f = f + b
```
```python
# With vectorization
  f = np.dot(w,x) + b # dot product of the 2 vectors
```
#### Gradient descent for multiple linear regression
- Parameters
  - w<sub>1</sub>,..., w<sub>n; b
- Model 
  - f<sub>vector w,b</sub>(vector x)= w<sub>1</sub>+...+w<sub>n</sub>+b
  - f<sub>vector w,b</sub>(vector x)= <vector>w</vector>.<vector>x</vector>+b
- Cost function
  - J(w<sub>1</sub>,...,w<sub>n</sub>,b)
- Gradient descent
 -  repeat 
    ```markdown
    {
        w(i) = w(i-1) - a (d/dw) J(w<sub>1</sub>,..., w<sub>n</sub>,b(i-1))
        b(i) = b(i-1) - a (d/db) J(w(w<sub>1</sub>,..., w<sub>n</sub>),b(i-1))
    } 
- How to make sure Gradient descent is working?
  - learning curve -> J(vector w, b) should go down
  - automatic convergence test
  - let epsilon (E) be 0.003
  - if in 1 iteration J(vector w,b) decreases by <= E, declare convergence (found parameterd vector w and b to get close to global minimum)
- Choosing a learning rate
  - from a scatter plot between iterations and J(vector w, b)
  - J(vector w, b) must go down with iterations
  - try values for alpha -> 0.001,0.01, 0.1, 1, ..
  - try alphas 3 times bigger than the previous one
  
#### Alternative to gradient descent only for linear regression
- Normal equation
  - slow when number of features is large (>10000)
  
### Feature scaling
- Range of parameters
- if the range of the parameter is large, use a smaller w<sub>n</sub>
- Scaling by a<= x<sub>1</sub> <= b; x<sub>1 scaled</sub> = x<sub>1</sub>/b
  - a/b <= x<sub>1 scaled</sub> <= 1
- scaled w ranges from 0 to 1
- Mean normalization
  - x<sub>1</sub> = (x<sub>1</sub> - u<sub>1</sub>)/ (b-a); where a<= x<sub>1</sub> <= b and u1 = mean(x<sub>1</sub>)
  - x<sub>2</sub> = (x<sub>2</sub> - u<sub>2</sub>)/ (b-a); where a<= x<sub>2</sub> <= b and  u2 = mean(x<sub>2</sub>)
- Z-score normalization
  - based on the standard deviation
  - x<sub>1</sub> = (x<sub>1</sub>-u<sub>1</sub>)/(SD<sub>1</sub>); where u1 = mean(x<sub>1</sub>) and SD<sub>1</sub>=SD(x<sub>1</sub>)
- with feature scaling; aim for about -1<=x<sub>j</sub><=1 for each j
- rescale only if the range of a feature is too large or too small



### Classification
Classifying an input based on multiple correctly previously classified inputs
- Malignant tumor vs benign tumor
- Can have multiple classficiation
- Output classes/ output categories
- Numeric/ non-numeric
- Small number of possible outputs unlike regression, which gives infinite number of outputs
- Can have two or more inputs
- False negative, false positives (likelihood)
- Parameters for classification


### Unsupervised learning
- Find something interesting/ pattern in unlabeled data
- Clustering -> adding multiple data into clusters
    - Finding common parameters and grouping them into articles
    - DNA microarray
    - Finding structure in data
    - Grouping customers
- Anomaly detection
    - To detect unusual data points
- Dimensionality reduction
    - jed
- Recommending systems
- Reinforcement learning

#### 
