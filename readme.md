### Machine learning algorithms
- Supervised learning
- Unsupervised learning
### Supervised learning
- give system mutliple right answers 
  based on multiple right answers system will detect the next rigth answer for an unanswered question
- Spam filtering
- Speech recognition
- Machine translation
- Online advertising
- Self-driving car
- Visual inspection

#### Regression
- Predict a number from infinitely many possibilities
- Can have infiite number of outputs for an input
- training set and testing set
- x as input (features) and y as output (target)
- m -> number of training examples (x,y); (x^i,y^i)
- n -> number of testing examples
- x -> f(x) -> Y; estimated -> Y-hat
- f(w,b)(x) = wx+b; f(x) = wx+b; b-> y intercept, x -> slope
 
##### Cost function
  - Parameters, weights, coefficients
  - Cost function
    - ![img.png](img.png)
  - Squared error cost function: Cost function divided by 2m to average the cost during to large tarining set
    - ![img_1.png](img_1.png)
  - Goal is to minimize the cost function J(w,b)
  - 

#### Classification
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
