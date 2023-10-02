# Machine Learning

Machine learning is the development of agents that are able to improve their performance criterion over time. More generally an agent is learning if it improves its performance when making observations about the world. Whan the agent is a computer it observes the world though data or examples and use it to build a model that acts an hypotesis and piece of software that can be solved.

Machine learning is needed when:

- Is not easy to express the problem in an algorithmic way.
- The problem as an infinite amount of solutions which cannot be predicted.

## Learn

In the context of machine learning we give this definition `learning` for computers :

$\text{A computer is said to learn from experience }E\text{ with respect to some class of task }T\text{ and performance measure }P,\\ \text{ if its performance at tasks in }T,\text{ measured by }P\text{ improves with experience }E.$

Is evident the importance of the experience $E$, in fact it can have a significant impact on success or failure of the learner.
In particular is very important how well $E$ represents the [distribution](#underlying-distribution) of examples over which the performance $P$ will be measured. In general, learning is most reliable when the training examples follow a [distribution](#underlying-distribution) similar to that of future examples.
In general however this is hardly the case and we have to resort to learning over a different [distribution](#underlying-distribution) over which the system will be avaluated. This is problematic because mastering a [distribution](#underlying-distribution) doesn't necesserarely lead to better performance over some other [distribution](#underlying-distribution).

In order to have theoretical results however [we need to make the assumption that the distribution of the training examples is the same of the test examples](./SupervisedLearning.MD/#the-inductive-learning-hypotesis), but is important to keep in mind that such assumption will often be implicitly violated.

## Target function

After deciding what is the task we want to solve, we need to determine what type of knowledge will be learned and how this will be used by the performance program.
The most obvious way of information to be learned is through a function called `target function`.
Is useful to reduce the original machine learning problem to the problem of improving the performance $P$ of the target function at task $T$, this gives us a clear explanation as to why how we choose the target function is a key design of a machine learning model.
The goal of learning is to discover an `operational` (computable in an efficient way, or simply computable in a reasonable time bound) description of the ideal target function.
In general this will not be possible and we'll have to settle for an approximation.

We now know that we need to specify $\hat{V}$, the target function, that best aproximies $V$ the ideal target function.
This means that now need to specify the rappresentation of $\hat{V}$ that best fits out problem.

## Machine learning problems

We can generalize a machine learning problem as:

$\text{The problem of learning a target function}:X\rightarrow{Y}\text{ given a training set }D\text{ containing information about }V$

In this context learning a [target function](#target-function) $V$ means computing an approximated function $\hat{V}$ that returns values as close as possible to $V$ in particular for samples that are not in the training set $D$.
We call $D$ the `dataset` of a specific machine learning problem, it contains instances of the experience $E$.
We can categorize machine learning problems depending on the type of dataset:

- **Supervised learning** : $D$ is the set of pairs of instances of $X$ and $Y$. We have to learn $V$ knowing $x\in{X}$ such that $V(x)=y\in{Y}$. Depending on the type of function that must be learned supervised learning problems divide into:

  - **Regression** : When the output set $Y$ is continuos.
  - **Classification** : When the output set $Y$ is discrete.

- **Unsupervised learning** : $D$ is the set instances of $X$. We have to learn $V$ without knowing any instance of $Y$.
- **Reinforcement learning** : $D$ is the set of executions (state, action and rewards). In this context we have to learn a behaviour.

## Underlying distribution

We are ofter interested in estimating the accuracy with which will classify future instances.
Suppose we have some space of possible instances $X$ over which various target functions are deifned. We also assume that different instances o $X$ may be encountered with different frequencies.
A convenient way to model this is to assume that there is some underlying unknown probability distribution $\mathcal{D}$, is important to note that the probability distribution informs us about what is the probability of encountering a given $x$ but it gives no informations about it. Given this assumption a learning task is to learn the target function or (target concept) by considering a space of possible hypotesis.
