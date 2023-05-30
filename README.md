# gibbs_mcmc_nn
Gibbs sampler for neural network posteriors.

1. `gibbs_mcmc_nn.py`contains all functions necessary for the Gibbs sampler to work
2. The notebook `mlp_bayes_optimal_regression.ipynb` provides an introduction to posterior sampling, the Gibbs sampler, and an implementation of the Gibbs sampler in the sace of a one hidden layer network, trained on synthetic data.
3. The notebook `mlp_gibbs_mnist.ipynb` runs the Gibbs sampler on a one hidden layer fully connected neural network with ReLU activations and MNIST as dataset.
4. The notebook `cnn_gibbs_mnist.ipynb` runs the Gibbs sampler on a simple cnn network, with MNIST as dataset.

For an introduction to this algorithm it is recommended to go through the notebooks at points 2,3,4 in this order.
