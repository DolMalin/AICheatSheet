# AICheatSheet
A non-exhaustive cheat sheet about what I learned in AI during the last months.

# Table of Contents
1. [Data Preparation](#data-preparation)
2. [Models](#models)
	1. [Machine Learning](#machine-learning)
	2. [Deep Learning](#deep-learning)
3. [Training](#training)
	1. [Loss functions](#loss-functions)
	
# Data Preparation
# Models
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
<a id="softmax-classifier"></a><details open>
<summary> Softmax Classifier</summary>

<img src="assets/images/mlmodels/softmaxclassifier.png" width=45% height=45%>

* Outputs a propabilistic interpretation *(due to softmax)*
	* All the outputs values of the function will be scaled between 0 and 1
* Uses a [Cross-Entropy Loss](#cross-entropy-loss)
* Similar results as [SVM](#svm)




</details>

## Deep Learning

---
# Training
## Loss functions

[//]: #hinge-loss
<a id="hinge-loss"></a><details>
<summary> Hinge Loss</summary>


* Used for training in **maximum-margin** classification.
* Known for being used in [Support Vector Machine](#svm) (SVM)
* $\ell(y) = max(0, 1 - t \cdot y)$
	* Where $t$ is the **actual outcome** *(either -1 or 1)* and $y$ is the **output** of the classifier. 
</details>


[//]: #cross-entropy-loss
<a id="cross-entropy-loss"></a><details open>
<summary> Cross-Entropy Loss</summary>

* Used in **binary** and **multiclass** classification
* **Entropy** means the average level of randomness or uncertainty.
* It measures the difference between **two probability distributions**:
	1. The discovered probability distribution of a ML classification model
	2. The predicted distribution
* Often compared to [log loss](#log-loss)
* **Binay** Cross-Entropy Loss:
	* $l = -(ylog(p) + (1 - y)log(1 - p))$
	* Where $p$ is the **predicted probability** and $y$ is the **actual outcome** (0 or 1)
* **Multiclass** Cross-Entropy Loss:
	* $-\sum_{c=1}^My_{o,c}\log(p_{o,c})$
	* Where $M$ is the **number of classes**
	* We calculate a separate loss for each label and sum the result.

</details>


[//]: #log-loss
<a id="log-loss"></a><details>
<summary> Log Loss</summary>
</details>