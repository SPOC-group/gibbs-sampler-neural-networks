# Gibbs sampler for neural networks' posteriors
This repository contains a MCMC algorithm to sample from the posterior distribution of neural networks. 
The algorithm is based on layer wise Gibbs sampling, i.e., at each step the algorithm fixes all variables except one and updates it by sampling from the corresponding conditional distribution.
The exact form of the posterior and the derivation of the algorithm can be found at https://arxiv.org/abs/2306.02729.


1. `gibbs_mcmc_nn.py`contains all functions necessary for the Gibbs sampler to work
2. The notebook `mlp_bayes_optimal_regression.ipynb` provides an introduction to posterior sampling, the Gibbs sampler, and an implementation of the Gibbs sampler in the case of a one hidden layer network, trained on synthetic data.
3. The notebook `mlp_gibbs_mnist.ipynb` runs the Gibbs sampler on a one hidden layer fully connected neural network with ReLU activations and MNIST as dataset.
4. The notebook `cnn_gibbs_mnist.ipynb` runs the Gibbs sampler on a simple cnn network, with MNIST as dataset.
5. The notebook `tests_gibbs.ipynb` contains the tests for the library. The tests are not automatic. Instead they consist of running some test cases in a notebok and verifying that certain observables behave as expected (for example some observables must be stationary and others must have a certain expected value known in advance)

For an introduction to this algorithm it is recommended to go through the notebooks at points 2,3,4 in this order.
