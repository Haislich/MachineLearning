# Supervised learning

## Introduction

### Hypotesis

As previously discussed we approximated [target function](./Introduction.MD/#target-function) $V$ with a function $\hat{V}$.
This was necessary because in order to learn $V$ we would need to know all $V(x)$, but we are missing both all possible instances and all possible results, this means that in general we have to settle with an approximation of $V$.

$\hat{V}$ has a special name, it's called an `hypotesis` $h$ and is defined as

$\text{The mathematical function or model taken from an hypotesis space }H\text{ (a vectorial space of functions)}.$

An hypotesis is chosen from an hypotesis space. But how do we determine the best hypotesis space?
In general prior knowledge about the problem is used, or we can also do exploratory data analysis.
If the previous methods fail to choose a good hypotesis space multiple hypotesis spaces are tested and evaluated and the best is choosen.

### Consistent hypotesis

Taking into account only the supervised learning problems, for which we know the value of $h(x)$, we can define `consistency` as:

$\text{An hypotesis }h\text{ is consistent with a dataset }D\text{ and target function }V\text{if and only if }h(x) = V(x), \forall{(x,h(x))\in{D}}$

### Version Space

The version space if the set of all consistent hypotesis with respect to $D$, more formally:

$\text{The version space }VS_{H,D}\text{ with respect to hypotesis space }H\text{ and training examples }D\text{, is the subset of hypotesis taken from }H\text{ consistent with the training examples in }D.$

### The inductive learning hypotesis

The goal of machine learning tasks is to find the best hypotesis $h$ such that predicts correct values $h(x')$ for $x'\notin{D}$.
Even if we want to approximate $V$ as best as we can, the only information that we have is that $V(x)=y\in{D},\forall{(x,y)\in{D}}$.
So the only guarantee that we have is that chosen a consistent hypotesis we will fit the data that we already have.
Lacking further information we make an assumption about the data.

$\text{Any hypotesis that approximates well over a sufficiently large set of training examples will allso approximate the target function well over the unobserved examples.}$

### Learning as a search

In essence, learning involves search: searching through a space of possible [hypotheses](#hypotesis) to find the hypothesis that best fits the available training examples and other prior constraints or knowledge.
It is important to note the by selecting an hypotesis representation, the designer of the learning algorithm implicitly defines the space of all hypotheses that the program can ever represent, and therefore can ever learn (a program would never learn anything that it cannot represent).
Viewing learning as a search problem means that we are particularly interessed in algorithms capable of efficienly search very large (or infinite) hypotesis spaces, in order to find the best fit for the training data.

### Concept learning

Generally speaking much of learning involves acquiring general concepts from specific training examples. The problem of automatically inferring the general definition of some concept given examples labeled as members or nonmembers of the concept is called `concept learning`.
Concept learning is the task of inferring a boolean valued function from training examples of its input and output.
Every instance of $X$ is a n-dimensional tuple in which $?$ means that every values is acceptable and $\emptyset{}$ means that no values is acceptable.

### Representation of an hypotesis

In concept learning we define a representation of an hypotesis as:
$\text{A representation of an hypotesis }h\text{ is a subset of the input space }X'\subseteq{X}\text{ in which } h(x) = True, \forall{x}\in{X'} $

### The list then eliminate algorithm

This algorithm finds all describable hypotesis that are consistent with the observed training examples, it finds the version space for $H$ and $D$. An obvious way to represent the version space is simply to list all of its members.
The idea behind this algorithm is first initialize the version space to contain all possible hypotesis and then through an iterative process eliminate all hypotesis that are not consistent.
In general we would like to see only one hypotesis as the output, but if not enough data is present to narrow down to a single hypotesis then the algorithm spits out the set of remaining hypotesis.
This algorithm works perfectly in practice because is guaranteed to output at least one consistent hypotesis.
It however requires exhaustively enumerating all hypotesis in $H$ and also that the hypotesis space is finite (thus countable). Which is unrealistic for most, if not all, real hypotesis spaces.

### Inductive Bias

The inductive bias is the set of assumptions that a machine learning algorithm makes about the relationship between iput variables and output variables.
In essence the deductive bias is the set of assumptions that, together with the training data, deductively justify the classifications assigned by the learner to future instances.
It answers the question about hte policy by which ID3 generalizes from observed training examples to classify unseen instances.

### Language bias and search bias

Suppose that we wish to assure that the hypotesis space contains the unknown target concept.
Suppose also that we use an [hypotesis representation](#representation-of-an-hypotesis) in which each hypotesis is a conjunction of costraints on the instance attributes.
Because of the restricion we posed on the hypotesis space, it is unable to represent the simplest disjunctive target concepts, thus posing a bias.
An obvious solution seems to have an hypotesis space capable of representing every teachable concept.
This means that every hypotesis in $H$ is mapped to a subset of $X$ for which the hypotesis is always True.
Let rephrase our task in an unbiased way by defining an hypotesis space that can represent every subset of instances, so the power set of $X$.
We can now be assured that no matter what the concept is, it is expressible within the given hypotesis space.
Unfortunately even if this new hypotesis space solves the problem of expressability, it raises a new problem, because the hypotesis space is unable to generalize beyond the seen examples.
This lack of generalization is due to the fact that since $H$ can represent the power set of $X$ given some unseen example $x$, there will be at least an hypotesis $h$ mapped to a subset of $X$ that doesn't represent $x$ ($h(x) = False$), but since $H$ can represent every subset of $X$ there must exist at least one hypotesis $h'$ that represents $x$ ($h'(x) = True$), this will cause indetermination and the machine learning model cannot give an answer.
In essence if the hypotesis space is enriched to the point where there is an hypotesis corresponding to every possible subset of instances, there will be no `inductive bias`, this lack of bias however completely removes the ability to classify new instances.
In general to avoid problems of indetermination we force a reduction of the expressive of the language, the process is called `language bias`.
As discussed in the [list then eliminate algorithm](#the-list-then-eliminate-algorithm) if not enough data is present to narrow down to a single hypotesis then the algorithm spits out the set of remaining hypotesis.
So in general is not guaranteed that there's a singular hypotesis, this can cause indetermination to happen, for the same reason.
So in general we make a choice and decide a particular ipothesis and we call this process `search bias`.

## Making evaluations

### Accuracy of an hypotesis

Evaluating the accuracy of an hypotesis is fundamental to machine learning, simply because in many cases it helps us to understand wheter to use the hypotesis or not.
When evaluating an hypotesis we are interested in estimatin ghe accuracy with which will classify new examples.
Statistical methods in conjunction with assumptions about the underlying distributions of data, allow one to give an evaluation of the perfomances of the hypotesis.
Estimating the accuracy of an hypotesis is pretty straightforward when data is plentiful.
However difficulties arises when we have a limited dataset, in particular as we'll see when we are dealing with a dataset we're implicitly dealing with a random variable.
This means that we have to deal with probabilities.

### Sample error, true error and accuracy

Before defining and understanding the differences between the two errors, we define the `sample data` $S$ as:

$\text{Sample of }n \text{ instances drawn from }X\text{ according to }\mathcal{D}\text{ for which }f(x)\text{ is known.}$

We oftesn assume implicitly that the elements of $S$ have this properties:

- The samples are identically distributed: the data-generating process is stable.
- The samples are idependently distribuited: there's no bias in the data collection, each sample is representative of the future and comprehensive of the past.

This two properties are costraints that we implicilty assume valid and stem from [the inductive learning hypotesis](#the-inductive-learning-hypotesis)

We call `sample error` the error rate of the hypotesis over the sample of data available.
More formally the sample error is:

$\\ \qquad error_s(h) = \frac{1}{n}\sum_{x\in{S}}\delta(f(x),h(x))$

Where $n$ is the number of examples in $S$ and

$\qquad\delta(f(x),h(x))= \begin{array}{ll}1 &\text{if }h(x) = f(x) \\0 &\text{otherwise.}\end{array}$

We call the `true error` the probability of misclassyfing a single randomly drawn instance from the distribution.
More formally:

$\qquad error_\mathcal{D} = Pr_{x\in{D}}[f(x)\neq h(x)]$

Where $Pr_{x\in{D}}$ is the probability to take a specific instance $x$ over $\mathcal{D}$.
This definition of the true error intrinsically assigns less value to misclassification of values that tend to appear less often.
Lastly we define `accuracy`:

$\qquad accuracy = 1-error(h)$

We usually want to know the true error, because is the error that we would expect when applying the hypotesis to future examples.
The problem is that we have no way of calculating it, this is because we have no idea on how the [underlying distribution](./Introduction.MD/#underlying-distribution) works, and also since $x$ is randomly drawn from there is no guarantee that $x\in{S}$, and for values outside of $S$ we don't know the value $f(x)$.

All we can measure is the sample error, keeping in mind that an high accuracy over the sample but we have low accuracy over the distribution our system will perform badly (we can correctly guess only what we already know).
The question is how well can we approximate the sample error to the true error?

### Error estimation

How does the deviation (a measure of difference between an observed value and some other value) between the sample error and the true erro depends on the size of the data?  This question is an instance of a  well-studied problem in statistics: the problem of estimating the proportion of a population that exhibits some property, given the observed proportion over some random  sample of the population. In  our  case, the property of interest is that $h$ misclassifies the example.
The key  to answering this  question is  to  note that  when we measure the sample error we are performing an experiment with  a random outcome. We first collect a random sample $S$ of $n$ independently drawn instances from the distribution $\mathcal{D}$, and then  measure the sample error $errors_S(h)$.
If we were to repeat this experiment many times we would expect to see different values for the various i $error_{s_i}(h)$, where $i$ is the $i$-th experiment,the outcome of such experiment would be called a random variable.
Imagine that we run $k$ different experiments and we measure the random variables $error_{S_1}(h)\dots error_{S_k}(h)$ and that we plot the histogram displaying the frequency with which we observe each possible value error. As $k$ grows we such histogram would approach the form of a binomial distriution.
The binomial distribution models the probability of observing $r$ errors in a data sample containint $n$ randomly and indipendent drawn instances.

### Estimating the true error

Statisticians calls $error_S(h)$ an estimator for $error_{\mathcal{D}}(h)$, an estimator is any random variable used to estimate some parameter (the probability $p$ of misclassifying any given instance) of ther underlying population from which the sample is drawn.
An obvious question is to ask yourself whether the estimator (the random variable) gives on average the right estimate (for the parameter we want to determine).
To answer this question we define the `estimation bias`, or simply `bias` as:

$\qquad bias = E[error_S(h)]-error_{\mathcal{D}}(h)$

If the bias is $0$ we say that $error_S(h)$ is a good approximation for $error_{\mathcal{D}}(h)$.
Is $error_S(h)$ an unbiased estimator for $error_{\mathcal{D}}(h)$ ?

Suppose that we have a probability $p$ of misclassyfing an input chosen at random ($error_{\mathcal{D}}(h) = p$).

$ E[error_S(h)] = E[\frac{1}{n}\sum_{x\in{S}}\delta(f(x),h(x))] \\
  \qquad \qquad \qquad = \frac{1}{n}\sum_{i=1}^{n} E[\delta(f(x_i),h(x_i))] \\
  \qquad \qquad \qquad = \frac{1}{n} \sum_{i=1}^{n} (1 * p + 0* (1-p)) \\
  \qquad \qquad \qquad = \frac{1}{n} np\\
  \qquad \qquad \qquad = p$

Where $E[\delta(f(x_i),h(x_i))] =  (1 * p + 0* (1-p))$ is given by the definition of the expected value, we either misclassify with probability $p$ or we don't misclassify with probability $1-p$.
This means that, in general, we don't make mistakes aproximating the true error by the sample error.
This is true only when $S$ and $h$ are choosen independently.
For this example let's call $D$ the full sample space that we have.
In order to choose $h$ and $S$ independently what we can do is split $D$ in two sets $S$ and $T$ such that $S\cap T= \empty \land T\approx\frac{2}{3}D$ (the value $\frac{2}{3}$ is purely empirical).
We then use $T$ to generate the target function and $S$ to evaluate the accuracy of $h$.
A note about the empirical $\frac{2}{3}$.
If $T$ is big we may improve the performance of the model, but we may lack generalization power.
If $S$ is too big we decrease variance, and the confidence interval of the estimator is low, however the value of the accuracy might not be satisfactory.
In general we make a trade off, but it depends on the specific application.

### Confidence Intervals

One common way to  describe the uncertainty associated with an estimate is to give an interval within  which the true  value is expected to  fall.

- The samples $S$ contains $n$ samples drawn indipendently from $\mathcal{D}$ and are independent of $h$
- $n\geq30$

from the statistical theory we can make the following assertions:

With an approximately $95\%$ the true error lies in the interval

$\qquad errors_S(h)\pm 1.96\sqrt{\frac{error_\mathcal{D}(1-errors_S(h))}{n}}$

### Overfitting

We say that an hypotesis $h\in{H}$ `overfits` the training data if the is there is an alternative hypotesis $h'\in{H}$ `overfits` if:

$\qquad error_S(h) < error_S(h') \land error_{\mathcal{D}}(h) > error_{\mathcal{D}}(h')$

In other $h$ overfits the data if has a smaller error than $h'$ in the training data but has greater error over the entire distribution of instances.
This is a really important problem, because it means that we're paying too much attention to the data that we already know and the hypotesis cannot generalize well.

### Evaluation of a learning algorithm

We're oftern interested in evaluating the performance bewteen two learning algorithms rather than two specific hypotesis.
Suppose we have two learning algorithms $L_A$ and $L_B$ and we want to determine which is better on average for learning some particular target function $f$.
A reasonable way to the fine on average is through the expected value, of the difference in their errors:

$\qquad E_{S\sub\mathcal{D}}[error_{\mathcal{D}}(L_A(S)) - error_{\mathcal{D}}(L_B(S))]$

Where $L(S) = h$.
In practive we have only a small sample set $D$, the approach is to divide it into a training set $T$ and a test set $S$, approzimate the sample error on $S$ with the true error and measure the difference:

$\qquad error_{S}(L_A(T)) - error_{S}(L_B(T))$

One way to improve this estimator is to repetedly partition in different disjoint training and test set and take the mean of these different experiments.
This procedure of partition multiple times the dataset is called K-fold cross validation.

### Performance metrics in classification

The question is, we're focusing on accuracy, but is it always a good performance metrics?
The answer is no, because imagine having two hypotesis $h_1$ and $h_2$ and an unbalanced dataset $D$ that contains $90\%$ of positive examples.
If the hypotesis $h_1$ have an accuracy of $80\%$ and is the result of a classification algorithm and $h_2$ has an accuracy of $90\%$ and is a function that returns the most common value in $D$, it is obvious that $h_2$ has a better accuracy than $h_1$ but this is misleading because ig the number of positive examples drop to $50\%$ the accuracy will also drop.
In general accuracy is not enough to asses the performance of a classification method.
We define:

- **True positive/TP** : A positive exampl correctly classified as positive
- **True negative/TN** : A negative example correctly classified as negative
- **False positive/FP**: A negative example classified as negative
- **False negative/FN**: A positive example classified as negative

And the following metrics:

- **Recall**: $\frac{TP}{TP+FN}$
- **Precision**: $\frac{TP}{TP+FP}$

The impact of false positives (Precision) and the impact of false negatives (Recall) is application dependant.
Imagine that you're creating an autonomous mobile and you're in charge of making the model that predicts positive and activates a brake system  if there's a pedestrian.
In this context having a false positive means: The system incorrectly guesses that there's a pedestrian and breaks, this may cause collision with other cars, but if the breaking system is correctly calibrated the risk of anyone gettin hurt should be minimal.
On the other hand having a false negative means: The system incorrectly guesses there is no pedestrian and doesn't break, causing the death/a severe damage of the pedestrian.
It's obvious that we want to maximize the recall.
However is not always straight forward to understand which needs to be maximized.
So in general we combine precision and recall in something called `f1-score`

$\qquad \text{F1-score}=2\frac{precision* recall}{precision+recall}$.

Lastly we also define something called the `confusion matrix` and is essentially a table containg every class $C_i$ thate we need to classify and computes how many times a class $C_i$ is misclassified as $C_j$, the main diagonal is the accuracy for each class and every cell outside the one in the diagonal counts the number of misclassification made.

## Decision Trees

Decision Tree learning is a method for approximating discrete-valued target functions, in which the learned function is represented by a decision tree.
Learned trees can also be re-represented as sets of if-then rules to improve human readability.

### Representation

Decision trees classify  instances by sorting them  down  the  tree  from the root to some leaf node, which provides the classification of the instance. Each node in the tree specifies a test of some attribute of the instance, and each branch descending
from  that  node corresponds to  one of the  possible values  for this  attribute.
In general, decision trees  represent  a disjunction of conjunctions of constraints (it can be represented as a set of If-Then statements)on the attribute values of instances.

### ID3

ID3 is the basic algorithm used for the construction of decision trees.
It learns the decision trees by constructing them  with a top-down approach.
The first thing it does is to create a root node.
Once the root is created asks the question "which attribute should be tested?".
If all instances are have the same value, the algorithm return the node with the correct label.
If the set of the attributes is empty the algorithm return the node with label the value of the most common instance.
Else the best attribute is selected (using statistical tecniques), used as test and removed from the set of attributes.
A descendant is then created for each possible value of this attribute and the training examples are sorted to the appropriate descendant node.
The entire process is then repeated  using  the  training examples associated with each descendant node to select the  best  attribute to test at that  point  in  the  tree.

The central part of this algorithm is the choice of the best atttribute to test at each node in the tree. We would like to select the attribute that is most useful for classifying examples.
A measure of usefuleness is given by the `information gain`, it measures how well a given attribute separates the training examples according to their target classification.
In order to define information gain we need to define another quantity called `entropy` which is a measure of impurity of the collection of examples.

$\qquad Entropy(S) = -p_{+}log_2(p_+)-p_{-}log_2(p_-)$

Where $p_+$ and $p_{-}$ is the probability of being positive and negative respectively.
Note that the previous definition of entropy only applies to a decision tree that performs binary classification.
If the entropy is $0$ all members belong to the same class, if the entropy if $1$ there's an equal member of positive and negative examples.
This idea of entropy can be of course generalized to not to just binary classification, but with multi-class classifictions with a total of $c$ different classes

$\qquad Entropy(S) = \sum_{i=1}^c-p_{i}log_2(p_i)$

Having defined entropy we can define the information gain as the expected reduction in entropy caused by partitioning the examples according to this attribute.
Given a collection of examples $S$ and an attribute $A$

$\qquad \text{Gain}(S,A) = \text{Entropy}(S) - \sum_{v\in \text{Values(A)}} \frac{|S_v|}{S}\text{Entropy}(S_v)$

Where $Values(A)$ is the set of all possible values for attribute $A$ and $S_v$ is the subset of $S$ for which attribute $A$ has value $v$.

### Hypotesis space search in decision tree learning

ID3 makes a search in the set of possible trees for a tree that fits the data.
ID3 starts with an empty tree and consider progressively more leaborate hypotesis in search of the correct tree.
The evaluation function that guides this process is the information gain measure.
By viewing ID3 in terms of search space we can get some informations about it's capabilities and limitations.
ID3 hypotesis space is a complete space of finite discrete-valued functions, relative to the available attributes, every finite discrete-valued function can be representd by some decision tree thus we can always create one.
ID3 maintain a single hypotesis and we don't have to reduce the language's power though a language bias, in turn sticking to a single hypotesis ID3 loses the ability to determine how many alternative decision trees are consistent with the data and resolve optimally the problem.
It never backtracks, so when an attribute is selected for test at a particular level it never reconsiders it choices, it is thus susceptible to converging to a local minima.
ID3 uses all training examples at each step to make decisions on how to refine the current hypotesis, this means that is much less sensitive to errors in indivisual training examples.
ID3 can also be extended to handle noisy data .

### Inductive bias in Decision tree learning

Given a collection of training examples, there are tipically many consistent trees associated with these examples.
Describing the inductive bias of ID3 consists of describing the basis over which it select an hypotesis over the others.
ID3 simply prefers short trees that places high information gain attributes near the root.
This inductive bias follows from ID3 search strategy and thus is called preference bias (or search bias).
For example however Candidate-Elimination algorithm bias is a restriction bias (language bias), this is because it searches completely an incomplete space, finding each consistent hypotesis with the trining data and the inductive bias is a consequence of the expressive power of its hypotesis representation.

### Issues in Decsion tree learning

The first problem that can arise is overfitting.
While it's in general a reasonable strategy to grow the tree to perfectly classify the training data, but can [overfit](#overfitting).
This can happen if for example training examples contains random errors or noise, this can cause ID3 to construct a more complex tree, searching for further refinements to adapt to the noise.
It can also happen when training data are noise-free and small numbers of examples are associated with leaf nodes.
In this case is ofter possible that coincidental regularities occur, in which some attribute happens to partition the examples very well, despite being unrelated to the actual target function.
In general overfitting is associated with the size of the tree, so two solutions are usually used to avoid overfitting.
The first is to stop growing the tree when the data split is not statistically significant, this approach is called early stopping, howver this approach is dangerous because it stops us from recognizing situations in which there are no good attributes to splits on but there are a combination of attributes that are informative.
Another approach is to let the tree overfit and then post-prune the tree.
Indipendently by the approach used we still need to determine a criterion to determine the correct final tree size.
The approaches include:

- Use a separate set of examples, distinct from the training examples to evaluate the utility of post-pruning.
- use all the available data for training, but appy a statistical test to estimate wether expansing or pruning a particular node.
- Use an explicit measure of complexity for encoding the exmaples and the decision trees.

The first approach is the most common and is often referred to as a training and validation set approach.
We first split data into training, used to form the hypotesis and validation set used to evaluate the accuracy and the impact of pruning.
The reason behind it is while the training set may misled the algorithm, the validation set is unlikely ti exhibit the same random fluctuations.
Using this approach is effective provided the provided data is large enough.
When data is limited however the evaluation set reduces the already not so large training set and thus the examples available for training.

<!-- ### Rule post-pruning

Is a succesfull method for fining an high accuracy hypotesis.
It is used by a variation ID3 called C4.5.
Rule post-pruning comprises the following steps:

1. Infer the decision tree from the training set until it fits well the training data, allowing it to overfit freely.
2. Convert the learned tree into an equivalent set of rules by creating one rule for each path from root node to leaf node
3. Prune (generalize) each rule indipendently
4. Sort final results into desired sequence for use. -->
