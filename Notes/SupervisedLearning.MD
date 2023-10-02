# Supervised learning

## Hypotesis

As previously discussed we approximated [target function](./Introduction.MD/#target-function) $V$ with a function $\hat{V}$.
This was necessary because in order to learn $V$ we would need to know all $V(x)$, but we are missing both all possible instances and all possible results, this means that in general we have to settle with an approximation of $V$.

$\hat{V}$ has a special name, it's called an `hypotesis` $h$ and is defined as

$\text{The mathematical function or model taken from an hypotesis space }H\text{ (a vectorial space of functions)}.$

An hypotesis is chosen from an hypotesis space. But how do we determine the best hypotesis space?
In general prior knowledge about the problem is used, or we can also do exploratory data analysis.
If the previous methods fail to choose a good hypotesis space multiple hypotesis spaces are tested and evaluated and the best is choosen.

## Consisten hypotesis

Taking into account only the supervised learning problems, for which we know the value of $h(x)$, we can define `consistency` as:

$\text{An hypotesis }h\text{ is consistent with a dataset }D\text{ and target function }V\text{if and only if }h(x) = V(x), \forall{(x,h(x))\in{D}}$

## Version Space

The version space if the set of all consistent hypotesis with respect to $D$, more formally:

$\text{The version space }VS_{H,D}\text{ with respect to hypotesis space }H\text{ and training examples }D\text{, is the subset of hypotesis taken from }H\text{ consistent with the training examples in }D.$

## The inductive learning hypotesis

The goal of machine learning tasks is to find the best hypotesis $h$ such that predicts correct values $h(x')$ for $x'\notin{D}$.
Even if we want to approximate $V$ as best as we can, the only information that we have is that $V(x)=y\in{D},\forall{(x,y)\in{D}}$.
So the only guarantee that we have is that chosen a consistent hypotesis we will fit the data that we already have.
Lacking further information we make an assumption about the data.

$\text{Any hypotesis that approximates well over a sufficiently large set of training examples will allso approximate the target function well over the unobserved examples.}$

## Learning as a search

In essence, learning involves search: searching through a space of possible [hypotheses](#hypotesis) to find the hypothesis that best fits the available training examples and other prior constraints or knowledge.
It is important to note the by selecting an hypotesis representation, the designer of the learning algorithm implicitly defines the space of all hypotheses that the program can ever represent, and therefore can ever learn (a program would never learn anything that it cannot represent).
Viewing learning as a search problem means that we are particularly interessed in algorithms capable of efficienly search very large (or infinite) hypotesis spaces, in order to find the best fit for the training data.

## Concept learning

Generally speaking much of learning involves acquiring general concepts from specific training examples. The problem of automatically inferring the general definition of some concept given examples labeled as members or nonmembers of the concept is called `concept learning`.
Concept learning is the task of inferring a boolean valued function from training examples of its input and output.
Every instance of $X$ is a n-dimensional tuple in which $?$ means that every values is acceptable and $\emptyset{}$ means that no values is acceptable.

<!-- ## General to specific ordering of hypotesis

Many algorithms for concept learning organize the [search](#learning-as-a-search) by relying on a very specific structure that exists for any concept learning problem: `general to specific ordering of hypotesis`. We can take advantage of this hypotesis to design learning algorithms that exhaustively search without explictly enumerating every hypotesis.
Assuming we have two hypotesis $h_1 = <\text{Sunny},?,?,\text{Strong}>$ and $h_2 = <\text{Sunny},?,?,?>$.
Because $h_2$ imposes fewer constrains on the instance than $h_1$, every instance classified positive byt $h_1$ will also be considered positive by $h_2$.
We can intuitively see that $h_2$ can be considered as "more general than" $h_1$.
We can define a `more general than or equal to` relation in terms of the sets of instances that satisfy two hypotesis.

$\text{Let }h_j\text{ and }h_k\text{ be boolean valued functions defined over } X\text{.}\\ \text{Then }h_j\text{ is more general than or equal to }h_k(h_j \ge_g h_k)\text{ if and oly if }\forall{x\in{X}}[(h_k(x)=1) \implies (h_j(x)= 1)]$

We find useful to extend this definition to `stricly more general`($h_j>_gh_k$) if and only if $(h_j \ge_g h_k) \land{} (h_k \ngeq_g h_j)$ and the inverse if we have $h_j \ge_g h_k$ we define $h_k$ as `more general than` $h_j$.

Formally the $\ge_g$ relation defines a partial order for the hypotesis space $H$.
Being a partial orderin it means that there could be two hypotesis $h_1$ and $h_2$ such that $h_1 \ngeq_g h_2$ and $h_2 \ngeq_g h_1$. -->

## Representation of an hypotesis

In concept learning we define a representation of an hypotesis as:
$\text{A representation of an hypotesis }h\text{ is a subset of the input space }X'\subseteq{X}\text{ in which } h(x) = True, \forall{x}\in{X'} $

## The list then eliminate algorithm

This algorithm finds all describable hypotesis that are consistent with the observed training examples, it finds the version space for $H$ and $D$. An obvious way to represent the version space is simply to list all of its members.
The idea behind this algorithm is first initialize the version space to contain all possible hypotesis and then through an iterative process eliminate all hypotesis that are not consistent.
In general we would like to see only one hypotesis as the output, but if not enough data is present to narrow down to a single hypotesis then the algorithm spits out the set of remaining hypotesis.
This algorithm works perfectly in practice because is guaranteed to output at least one consistent hypotesis.
It however requires exhaustively enumerating all hypotesis in $H$ and also that the hypotesis space is finite (thus countable). Which is unrealistic for most, if not all, real hypotesis spaces.

## Language bias and search bias

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

## Accuracy of an hypotesis

Evaluating the accuracy of an hypotesis is fundamental to machine learning, simply because in manyn cases it helps us to undertand wheter to use the hypotesis or not.
When evaluating an hypotesis we are interested in estimatin ghe accuracy with which will classify new examples.
Statistical methods in conjunction with assumptions about the underlying distributions of data, allow one to give an evaluation of the perfomances of the hypotesis.
Estimating the accuracy of an hypotesis is pretty straightforward when data is plentiful.
However difficulties arises when we have a limited dataset.

## Sample error and true error

Before defining and understanding the differences between the two errors, we define the `sample data` $S$ as:

$\text{Sample of }n \text{ instances drawn from }X\text{ according to }\mathcal{D}\text{ for which }f(x)\text{ is known.}$

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
We usually want to know the true error, because is the error that we would expect when applying the hypotesis to future examples.
The problem is that we have no way of calculating it, this is because we have no idea on how the [underlying distribution](./Introduction.MD/#underlying-distribution) works, and also since $x$ is randomly drawn from there is no guarantee that $x\in{S}$, and for values outside of $S$ we don't know the value $f(x)$.
