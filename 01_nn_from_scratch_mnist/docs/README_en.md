# MNIST Digit Recognition: Full Formula Derivation & Pure Manual Backpropagation

[English](./README_en.md) | [简体中文](./README.md)

> Targeted at Machine Learning beginners: From network architecture, forward propagation, and cross-entropy loss, to matrix calculus, the chain rule, and a complete hand-derived backpropagation.
> Demystifying the math behind single/multi-layer neural networks step-by-step without relying on black-box frameworks. Includes dimensionality checks and executable Python pseudocode.

> 🚀 **Want to run the code directly?** The complete pure NumPy implementation corresponding to this guide (including the basic single-sample version and the practical Mini-batch version) is ready in the [../code](../code) directory. Head over to check out executable scripts like [mnist_numpy.py](../code/mnist_numpy.py)!

**Target Audience:**
- Backend/Full-stack developers looking to bridge the gap in low-level neural network mathematics.
- Students getting started with algorithms, deep learning, and convex optimization.
- Job seekers preparing for interviews involving neural network derivations and gradient-based algorithm questions.

## Reading Guide
If you are a beginner, it is recommended to read the "Prerequisites" first to ensure you have the foundational knowledge, then refer to the "Notation Cheat Sheet" to understand the dimensionality of each formula. For the derivation part, focus on mastering the "Universal Laws" before applying them layer by layer. For code implementation, you can refer to the pseudocode at the end.

## Prerequisites:
- Basic neural network architecture
- Matrix multiplication
- Calculus (Partial derivatives)
- Multivariate calculus chain rule (Vector-to-vector derivatives)
- Gradient descent algorithm

### 💡 Notation Cheat Sheet
*Before diving into matrix calculus, please use this table as your "navigation map". Come back to check dimensions at any time during the derivation.*

| Notation | Meaning | Dimension Example |
| :--- | :--- | :--- |
| $x$ | Flattened column vector of the input image | $784 \times 1$ |
| $h_i$ | Linear output of the $i$-th layer (pre-activation) | Depends on the layer |
| $a_i$ | Activation output of the $i$-th layer | Same as above |
| $W_i$ | Weight matrix of the $i$-th layer | e.g., $16 \times 784$ |
| $b_i$ | Bias vector of the $i$-th layer | $16 \times 1$ |
| $J_i$ | Jacobian matrix of the $i$-th layer (derivative of output w.r.t input) | Depends on the transformation |
| $g_v$ (e.g., $g_h, g_i$) | $\frac{\partial L}{\partial v}$, gradient of the loss w.r.t a node variable $v$ (for error propagation) | Same as the corresponding node |
| $g_W, g_b, g_X$ | $\frac{\partial L}{\partial W}$ etc., gradient of the loss w.r.t weights, biases, and inputs | Same as the corresponding variable |
| $p$ | Probability vector output from Softmax | $10 \times 1$ |
| $y$ | One-Hot true label vector | $10 \times 1$ |
| $L$ | Cross-Entropy Loss | Scalar |
| $\alpha$ | Learning rate | Scalar |
| $\odot$ | Element-wise multiplication (Hadamard Product) | Vectors/Matrices of the same dimension |

## 1. Network Architecture & Forward Propagation

![Neural Network Architecture](../assets/神经网络模型图.svg)

Assuming the input image is flattened into a column vector $x$, the network consists of two hidden layers and one output layer.

- **Input Layer**
  - Input variable: $x$
  - Vector dimension: $(784 \times 1)$
- **Hidden Layer 1**
  - Linear computation: $h_1 = W_0 x + b_0$
  - Activation function: $a_1 = \text{ReLU}(h_1)$
  - Parameter dimensions: $W_0$ is $(16 \times 784)$, $b_0$ is $(16 \times 1)$, $a_1$ is $(16 \times 1)$
- **Hidden Layer 2**
  - Linear computation: $h_2 = W_1 a_1 + b_1$
  - Activation function: $a_2 = \text{ReLU}(h_2)$
  - Parameter dimensions: $W_1$ is $(15 \times 16)$, $b_1$ is $(15 \times 1)$, $a_2$ is $(15 \times 1)$
- **Output Layer**
  - Linear computation: $h_3 = W_2 a_2 + b_2$
  - Activation function: $p = \text{Softmax}(h_3)$
  - Probability formula: $p_i = \frac{e^{h_{3,i}}}{\sum_{k=1}^{10} e^{h_{3,k}}}$
  - Parameter dimensions: $W_2$ is $(10 \times 15)$, $b_2$ is $(10 \times 1)$, $p$ is $(10 \times 1)$

Note: The mathematical definition of the ReLU activation function used in the hidden layers is $\text{ReLU}(h) = \max(0, h)$. This operation applies independently to each element in the vector (i.e., if an element is less than 0, it is set to 0; if greater than 0, it remains unchanged). Note that ReLU is mathematically non-differentiable at $h=0$. In engineering implementations, the concept of a subgradient is usually applied, artificially assigning its derivative at $h=0$ to either 0 or 1 (assigned to 0 in this code implementation).

*To simplify the derivation, a smaller number of neurons (16 and 15) is used for the hidden layers. In real-world applications, this can be scaled up to 128 or 256.*

## 2. Loss Function
For multi-class classification problems, we use the Cross-Entropy Loss function.

$$
L = -\sum_{k=1}^{10} y_k \ln(p_k)
$$

Note: $y$ is the One-Hot column vector of the true label, with dimension $(10 \times 1)$. If the correct class is $c$, then $y_c = 1$, and the rest $y_{i \neq c} = 0$.

## 3. Backward Propagation
The core logic of backpropagation is divided into two steps: first, calculating the error transmission between nodes (node gradient $g$), and then using the node gradients to calculate the updates for the weights and biases of that layer.

### Supplement: Two Universal Laws of Neural Network Backpropagation

In neural networks, the most core computations alternate between adjacent layers: **Linear Transformations** and **Activation Functions**. We can distill the formulas for these two operations into a highly symmetrical forward-backward contrast format. (Define $g$ as the gradient of the loss function $L$ with respect to the current variable).

The core formula of the multivariate chain rule:

$$
\bar{g}_i = J_i^T \bar{g}_{i+1}
$$

Where $J_i$ represents the Jacobian matrix of the current layer.

### Law 1: Propagation Rule for Linear Transformation Layers (Simplified Jacobian)
Assume a linear transformation from input node $X$ through weight matrix $W$ to get the output node $h$. The forward computation formula is:

$$h = WX + b \quad [\text{Forward Propagation}]$$

Variables and typical dimension descriptions:
- $W$: Weight matrix of the current layer, assuming dimension is $(15 \times 16)$.
- $X$: Represents the input of the current layer (activation output of the previous layer, column vector), assuming dimension is $(16 \times 1)$.
- $b$: Bias term of the current layer (column vector), assuming dimension is the same as $h$, which is $(15 \times 1)$.
- $h$: Linear output node of the current layer (column vector), its dimension is $(15 \times 1)$.
- $g_h$: The gradient passed back from the next layer, representing the gradient of the loss $L$ w.r.t the output $h$ $\left(\frac{\partial L}{\partial h}\right)$. Its dimension is the same as $h$, which is $(15 \times 1)$.

Based on the core formula of the multivariate chain rule, we can directly derive three ultimate derivative conclusions for $X$, $W$, and $b$:

#### Conclusion 1: Gradient w.r.t input node $X$ (Used for passing the error further backward)
When deriving with respect to the input node $X$, the Jacobian matrix $J$ here is exactly the weight $W$. The gradient of the current layer's node equals the transpose of the weight matrix multiplied by the gradient of the next layer:

$$
g_X = \frac{\partial L}{\partial X} = W^T g_h \quad [\text{Backward Propagation}]
$$

- Dimension check: $(16 \times 15) \times (15 \times 1) = (16 \times 1)$, perfectly matching the original dimension of $X$.

#### Conclusion 2: Gradient w.r.t weights $W$ (Used for updating current layer parameters)
When deriving with respect to the weight $W$, the corresponding Jacobian matrix relates to the input node $X$. The formula for calculating the gradient of the weight matrix is the outer product of the next layer's gradient column vector and the previous layer's input row vector:

$$
g_W = \frac{\partial L}{\partial W} = g_h X^T \quad [\text{Backward Propagation}]
$$

- Dimension check: $(15 \times 1) \times (1 \times 16) = (15 \times 16)$, perfectly matching the original dimension of $W$.

#### Conclusion 3: Gradient w.r.t bias $b$ (Used for updating current layer parameters)
When deriving with respect to the bias $b$, since the partial derivative of the addition operation is 1 (the Jacobian matrix is the identity matrix $I$), the gradient of the bias is exactly equal to the gradient passed back from the next layer:

$$
g_b = \frac{\partial L}{\partial b} = g_h \quad [\text{Backward Propagation}]
$$

- Dimension check: $g_b$ directly inherits the dimension of $g_h$, which is $(15 \times 1)$, perfectly matching the original dimension of $b$.

### Law 2: Propagation Rule for Activation Function Layers
Assume the pre-activation input node is $h$, apply the activation function $\Phi(\cdot)$ (like ReLU) to get the output node $a$.

$$
a = \Phi(h) \quad [\text{Forward Propagation}]
$$

According to the multivariate chain rule, the derivative formula when the error penetrates the activation function is:

$$
g_h = g_a \odot \Phi'(h) \quad [\text{Backward Propagation}]
$$

- **Notation explanation**: Both the activation function $\Phi(\cdot)$ and its derivative $\Phi'(\cdot)$ act independently on each element of the vector. The symbol $\odot$ denotes **Element-wise Multiplication (Hadamard Product)**.

*Once you master these two universal laws, the complex derivations for the subsequent hidden layers are simply alternating combinations of these two formulas! -- Detailed derivations to be updated.*

### 3.1 Output Layer Gradients
Calculate the derivative of the loss $L$ with respect to the output layer's pre-activation value $h_3$, denoted as $g_3$. Mathematically, $g_3 = \left( \frac{\partial p}{\partial h_3} \right)^T \frac{\partial L}{\partial p}$. Through the chain rule derivation combining Cross-Entropy and Softmax, we get an extremely simple subtraction form:

$$
g_3 = \frac{\partial L}{\partial h_3} = p - y
$$

- Dimension check: $g_3$ is a $(10 \times 1)$ column vector.

*This result is the famous simplified form of combining Cross-Entropy loss + Softmax activation. Detailed derivation to be updated.*

Calculate the gradients for the 3rd layer's weights and biases:

$$
\frac{\partial L}{\partial W_2} = g_3 a_2^T
$$

$$
\frac{\partial L}{\partial b_2} = g_3
$$

- Dimension check: $\frac{\partial L}{\partial W_2} = (10 \times 1) \times (1 \times 15) = (10 \times 15)$, perfectly matching the original dimension of $W_2$.

### 3.2 Hidden Layer 2 Gradients
The error propagates from $h_3$ back to $h_2$. It needs to be multiplied by the transpose of the weight matrix and pass through the ReLU activation function (using element-wise multiplication $\odot$):

$$
g_2 = \frac{\partial L}{\partial h_2} = \left( W_2^T g_3 \right) \odot \text{ReLU}'(h_2)
$$

- Note: $\text{ReLU}'(h_2)$ is the derivative vector. Its elements are $1$ when $h_2 > 0$, otherwise $0$.
- Dimension check: $W_2^T$ is $(15 \times 10)$, $g_3$ is $(10 \times 1)$. The result of their matrix multiplication is $(15 \times 1)$. After element-wise multiplication with the ReLU derivative, $g_2$ remains a $(15 \times 1)$ column vector.

Calculate the gradients for the 2nd layer's weights and biases:

$$
\frac{\partial L}{\partial W_1} = g_2 a_1^T
$$

$$
\frac{\partial L}{\partial b_1} = g_2
$$

- Dimension check: $\frac{\partial L}{\partial W_1} = (15 \times 1) \times (1 \times 16) = (15 \times 16)$, perfectly matching the original dimension of $W_1$.

### 3.3 Hidden Layer 1 Gradients
Similarly, the error propagates from $h_2$ back to $h_1$:

$$
g_1 = \frac{\partial L}{\partial h_1} = \left( W_1^T g_2 \right) \odot \text{ReLU}'(h_1)
$$

- Dimension check: $W_1^T$ is $(16 \times 15)$, $g_2$ is $(15 \times 1)$, and after matrix multiplication, $g_1$ is a $(16 \times 1)$ column vector.

Calculate the gradients for the 1st layer's weights and biases:

$$
\frac{\partial L}{\partial W_0} = g_1 x^T
$$

$$
\frac{\partial L}{\partial b_0} = g_1
$$

- Dimension check: $\frac{\partial L}{\partial W_0} = (16 \times 1) \times (1 \times 784) = (16 \times 784)$, perfectly matching the original dimension of $W_0$.

## 4. Parameter Update
Using the collected gradients, update all weights and biases synchronously using the learning rate $\alpha$ (Gradient Descent):

$$
W_2 \leftarrow W_2 - \alpha \frac{\partial L}{\partial W_2}
$$

$$
W_1 \leftarrow W_1 - \alpha \frac{\partial L}{\partial W_1}
$$

$$
W_0 \leftarrow W_0 - \alpha \frac{\partial L}{\partial W_0}
$$

$$
b_2 \leftarrow b_2 - \alpha \frac{\partial L}{\partial b_2}
$$

$$
b_1 \leftarrow b_1 - \alpha \frac{\partial L}{\partial b_1}
$$

$$
b_0 \leftarrow b_0 - \alpha \frac{\partial L}{\partial b_0}
$$

*Note: Parameter updates should be done uniformly (synchronously) only after all gradient calculations are completed. Otherwise, using already updated weights will affect the correctness of subsequent gradients.*

### Pseudocode

```python
# Note: Using numpy matrix operation conventions (@ for matrix multiplication, * for element-wise multiplication)

# Forward Propagation
h1 = W0 @ x + b0
a1 = relu(h1)
h2 = W1 @ a1 + b1
a2 = relu(h2)
h3 = W2 @ a2 + b2
p = softmax(h3)
L = cross_entropy(y, p)

# Backward Propagation
g3 = p - y
dW2 = g3 @ a2.T
db2 = g3

g2 = (W2.T @ g3) * relu_deriv(h2)
dW1 = g2 @ a1.T
db1 = g2

g1 = (W1.T @ g2) * relu_deriv(h1)
dW0 = g1 @ x.T
db0 = g1

# Parameter Update
W2 -= alpha * dW2; b2 -= alpha * db2
W1 -= alpha * dW1; b1 -= alpha * db1
W0 -= alpha * dW0; b0 -= alpha * db0

```

## 5. Complete Derivations (To be updated)
### 5.1 Complete Proof of Combined Derivative for Softmax + Cross-Entropy Loss
Proof: $\displaystyle \frac{\partial L}{\partial h_3} = p - y$

### 5.2 Detailed Explanation of the ReLU Activation Function Derivative

### 5.3 Batch Sample Extension Derivation (Common in Engineering Practice)
This document provides the derivation for a **single sample**. In actual training, batch data is used, which introduces means and dimensional broadcasting. Batch version formulas will be supplemented later.

## 6. Extended Learning Path
Building upon this derivation, here are directions for extended learning:
1. Convex Optimization Basics (Advanced gradient descent, constrained optimization)
2. Backpropagation for various activation functions (Sigmoid/Tanh)
3. Mathematical derivation of Regularization (L1/L2) and Dropout
4. Optimizers: SGD and Adam principles and formulas