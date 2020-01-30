# Hypercomplex-Valued Recurrent Correlation Neural Networks

Hypercomplex numbers generalize the notion of complex and hyperbolic numbers as well as quaternions, tessarines, octonions, and many other high-dimensional algebras including Clifford and Cayley-Dickson algebras. This repository contains the source-codes and examples of a several hypercomplex-valued exponential correlation neural networks (RCNNs), which include the bipolar, complex-valued, and quaternionic ECNN models as particular instances.

## Getting Started

This repository contain the Julia source-codes of some hypercomplex-valued ECNNs, as described in the paper "Hypercomplex-Valued Recurrent Correlation NeuralNetworks" by Marcos Eduardo Valle and Rodolfo Lobo. The Jupyter-notebook of the computational experimens are also available in this repository.

## Usage

First of all, call the QRPNN module using:

```
include("HyperECNNs.jl")
```

### Bilinear Form and Activation Function

Effective use of hypercomplex-valued ECNN models requires a bilinear form and an appropriate hypercomplex-valued activation functions. Precisely, to take advantages of matrix operations, the hypercomplex-valued ECNN a function called BilinearForm whose inputs are a hypercomplex-valued matrix U organized in an array of size NxnxP, a hypercomplex-valued vector x organized in an array of size Nxn, and a set of parameters:
```
y = BilinearForm(U,x,Params)
```
Here, N denotes the the length of the vectors while n is the dimension of the hypercomplex numbers. For example, n=2 for complex numbers and n=4 for quaternions. The output y is such that y_i = \sum_{i=1}^N B(u_{ij},x_j), where B denotes the bilinar form. An example of the bilinar form is obtained using the command:
```
BilinearForm = HyperECNNs.LambdaInner
```
In this function, the bilinear form coincides with a weighted inner product.

In a similar fashion, the activation function is defined as follows where x is a hypercomplex-valued vector organized in an array of size Nxn and ActFunctionParams is a set of parameters:
```
y = ActFunction(x,ActFunctionParams)
```
Examples of activation functions in the HyperECNNs module include:
```
HyperECNNs.csign, HyperECNNs.twincsign, and HyperECNNs.SplitSign
```
See the reference paper for details.

### Hypercomplex-Valued Exponential Correlation Neural Networks

The module contrains two different implementations of a hypercomplex-valued ECNN. One using synchronous update and the other using sequential (or asynchronous) update. Stability is ensured using both update modes. Synchronous and sequencial hypercomplex-valued ECNNs are called respectively using the commands:
```
y, Energy = HyperECNNs.Sync(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
```
and
```
y, Energy = HyperECNNs.Seq(BilinearForm, BilinearFormParams, ActFunction, ActFunctionParams, U, xinput, alpha = 1, beta = 0, it_max = 1.e3)
```
Here, U is the hypercomplex-valued matrix whose columns correspond to the fundamental memories and organized in a real-valued array of size NxnxP (U[:,:,1] corresponds to the first fundamental memory) and xinput is a hypercomplex-valued vector organized in a real-valued array of size Nxn. The parameter alpha and beta define the excitation function f(x) = beta exp(alpha x). Finally, it_max specifies the maximum number of iterations.

See examples in the Jupyter notebook files.

## Authors
* **Marcos Eduardo Valle and Rodoldo Lobo** - *University of Campinas*
