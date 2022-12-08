# AICheatSheet
A non-exhaustive cheat sheet about what I learned in AI during the last months.

# Table of Contents
1. [Data](#data)
2. [Models](#models)
	1. [Components](#components)
		1. [Activation Functions](#activation-functions)
	2. [Machine Learning](#machine-learning)
	3. [Deep Learning](#deep-learning)
3. [Training](#training)
	1. [Loss functions](#loss-functions)
	
# Data
*@TODO*
# Models
## Components
### Activation Functions
[//]: #softmax
<a id="softmax"></a><details>
<summary>Softmax</summary>

* Convert a vector of real numbers into a *probability distribution* of these outcomes
* Often used as the last [activation function](#activation-functions) of a Neural Network to normalize ouptut as probabilities
* Often paired with **argmax** function that permits to get the highest probability
* Where $z$ is the vector of *raw outputs* from the Neural Network, $K$  is the number of classes.
	* We divide the exponential of one element of the output to the sum of all exponentials values of the output vector.
$$\sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}}$$
</details>


[//]: #relu
<a id="relu"></a><details>
<summary>Rectified Linear Unit (ReLU)</summary>

*@TODO*
</details>


[//]: #tanh
<a id="tanh"></a><details>
<summary>TanH</summary>

*@TODO*
</details>


[//]: #sigmoid
<a id="sigmoid"></a><details>
<summary>Sigmoid</summary>

*@TODO*
</details>

## Machine Learning

[//]: #KNN
<a id="knn"></a><details>
<summary>k-Nearest Neighbor (kNN)</summary>

<img src="assets/images/mlmodels/knn_concept.jpg" width=50% height=50%>

* It classifies a new entry by assigning it to the class of its closests neighbors.
* $k$ is the number of neighbors (datapoints) to compare to our new data point.
* It's a **non-parametric** approach
* When $k = 1$ *(also called 1-nearest neighbor)*
	* The algorithm will always achieve a training error of **zero**.
	* The algorithm is **consistent** *(eventually converging to the optimal predictor)*
* Requires to specify **distance function** $d$:
	* **Euclidian Distance** is the most popular
</details>


[//]: #SVM
<a id="svm"></a><details>
<summary> Support Vector Machine and Multiclass Support Vector Machine (SVM)</summary>

<img src="assets/images/mlmodels/svm.png" width=30% heightsoftmax-classifier=30%>

* It tries to find a line that **maximises** the separation between a **two-class** (SVM) or more (Multiclass SVM) dataset.
* The datapoints with the minimum distance to the hyperplane are called **Support Vectors**.
* Requires to specify a **kernel function** to compute datapoint separation:
	* Linear
	* Polynomial
	* Gaussian
	* Sigmoid
	* Radial Basis Function (RBF)

<img src="assets/images/mlmodels/svmkernels.webp" width=40% height=40%>	

*These functions will determine the smoothness and efficiency of class separation.*
* Use the [hinge loss](#loss-functions) to threshold the result to 0 if the correct score is greater than the incorrect class score by at least the margin.
	* The SVM only cares if the difference is lower than the margin $\Delta$
* Similar results as [Softmax Classifier](#softmax-classifier)

</details>


[//]: #softmax-classifier
<a id="softmax-classifier"></a><details>
<summary> Softmax Classifier</summary>

<img src="assets/images/mlmodels/softmaxclassifier.png" width=45% height=45%>

* Outputs a **propabilistic** interpretation *(due to [softmax](#softmax))*
	* All the outputs values of the function will be scaled between 0 and 1
* Uses a [Cross-Entropy Loss](#cross-entropy-loss)
* Similar results as [Support Vector Machine](#svm) (SVM)
* Provides kind of probabilities that are easier to interpret than SVM.

</details>

## Deep Learning
*@TODO*

---
# Training
## Loss functions

[//]: #hinge-loss
<a id="hinge-loss"></a><details>
<summary> Hinge Loss</summary>

* Also known as **maximum-margin loss**.
* Used for training in **maximum-margin** classification.
* Known for being used in [Support Vector Machine](#svm) (SVM)
$$\ell(y) = max(0, 1 - t \cdot y)$$
* Where $t$ is the **actual outcome** *(either -1 or 1)* and $y$ is the **output** of the classifier. 
</details>


[//]: #cross-entropy-loss
<a id="cross-entropy-loss"></a><details>
<summary> Cross-Entropy Loss</summary>

* Used in **binary** and **multiclass** classification
* **Entropy** means the average level of randomness or uncertainty.
* It measures the difference between **two probability distributions**:
	1. The discovered probability distribution of a ML classification model
	2. The predicted distribution
* Often compared to [log loss](#log-loss)
* **Binary** Cross-Entropy Loss:
$$l = -(ylog(p) + (1 - y)log(1 - p))$$
* Where $p$ is the *predicted probability* and $y$ is the *actual outcome* (0 or 1)
* **Multiclass** Cross-Entropy Loss:
$$l =-\sum_i^C y_i log(p_i)$$
* Where $y_i$ is the *actual outcome*, $p_i$ is the *predicted probability* of the $i^{th}$ label, and $C$ the *number of classes*
* We calculate a separate loss for each label and sum the result.

</details>


[//]: #log-loss
<a id="log-loss"></a><details>
<summary> Log Loss</summary>

*@TODO*
</details>