# BayesianDL

## Motivation
Deep Learning models are not infallible yet, and their reliability is crucial in applications such as medical diagnosis and autonomous driving, where the consequences of model's mistakes can be fatal. Therefore, measuring the uncertainty of Deep Learning models is essential to ensure their safety and reliability. There are two types of uncertainty that are typically considered: **epistemic uncertainty** refers to the model uncertainty that arises from a lack of knowledge or information about the system being modeled and **aleatoric uncertainty** which comes from random variations or noise in the system.


One approach for incorporating uncertainty in Deep Learning models is **Bayesian Deep Learning (BDL)**. The main altenative to incorporate aleatoric uncertainty in Neural Networks (NN) is to add a "head" at the end of the network to predict the variance (two ''heads'' BNN). Hence, the predicted output variance is input-dependent.





Regarding epistemic uncertainty, the main BDL techniques that aim to tackle this problem are: 

- **Ensembles**: train N models with different architectures, hyperparameters, or initial weights [^1]. As result, you may combine their predictions to produce a more accurate final prediction -i.e average among model predictions- and uncertainty estimation -i.e variance among model predictions.


- **MC Dropout**: extrapolates Dropout regularization technique to test time [^2]. It performs several forward passes randomly dropping out different hidden units during each one. As result, it generates multiple predictions for a given input, which can be used to estimate the model's uncertainty as before.


- **Laplace**: Laplace Approximation, originally introduced by David Mackay in 1992 [^3], has gained increasing attention in recent years. It approximates the posterior distribution of model's parameters through a Gaussian distribution, allowing for inference and avoiding overconfidence. However, the main bottleneck of this technique comes from computation and memory resources.


## Contribution
Here, we provide some notebooks in which we compare these approaches leveraging two Regression datasets: a Simulated dataset and Boston dataset. The Simulated dataset was designed to describe the meaning of both aleatoric and epistemic uncertainty and, hence, how the BDL techniques should estimate them. We compare them in terms of accuracy (MAE), calibration (AUSE) and likelihood (NLL). Specifically, to implement Laplace approximation we leverage the library created by Immer et. al .

## Bibliography
[^1]: \cite{lakshminarayanan2017simple}
[^2]: \cite{gal2016dropout}
[^3]: \cite{mackay1992bayesian}
