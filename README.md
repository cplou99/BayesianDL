# BayesianDL
![.](/Images/BDLIntro.PNG)

## Motivation
Deep Learning (DL) models are not infallible yet, and their reliability is crucial in applications such as medical diagnosis and autonomous driving [^1], where the consequences of model's mistakes can be fatal. 
![.](/Images/BDLMotivation.PNG)

Therefore, measuring the uncertainty of Deep Learning models is essential to ensure their safety and reliability [^2]. There are two types of uncertainty that are typically considered: **epistemic uncertainty** refers to the model uncertainty that arises from a lack of knowledge or information about the system being modeled and **aleatoric uncertainty** which comes from random variations or noise in the system.

## Estimating uncertainty in DL
### Aleatoric uncertainty
One approach for incorporating uncertainty in Deep Learning models is **Bayesian Deep Learning (BDL)**. The main altenative to incorporate aleatoric uncertainty in Neural Networks (NN) is to add a "head" at the end of the network to predict the variance (two ''heads'' BNN). Hence, the predicted output variance is input-dependent (heterocedastic) [^3].
![](/Images/OnevsTwoHeadsBNN.PNG)

### Epistemic uncertainty
Regarding epistemic uncertainty, the main BDL techniques that aim to tackle this problem are: 

- **Ensembles**: train N models with different architectures, hyperparameters, or initial weights [^4]. As result, you may combine their predictions to produce a more accurate final prediction -i.e average among model predictions- and uncertainty estimation -i.e variance among model predictions.


![.](/Images/Ensembles.PNG)


- **MC Dropout**: extrapolates Dropout regularization technique to test time [^5]. It performs several forward passes randomly dropping out different hidden units during each one. As result, it generates multiple predictions for a given input, which can be used to estimate the model's uncertainty as before.


![.](/Images/MC-Dropout.PNG)

- **Laplace**: Laplace Approximation, originally introduced by David Mackay in 1992 [^6], has gained increasing attention in recent years [^7], [^8]. It approximates the posterior distribution of model's parameters through a Gaussian distribution, allowing for inference and avoiding overconfidence. More detailly,
$$p(\pmb{\theta}|\mathcal{D}) = \frac{p(\mathcal{D}|\pmb{\theta})p(\pmb{\theta})}{p(\mathcal{D})} = \frac{1}{Z}g(\pmb{\theta}).$$
Then, it approximates $g(\pmb{\theta})$ following next steps,


![.](/Images/Laplace.PNG)


As result, it gets,
$$p(\pmb{\theta}|\mathcal{D}) \approx  \mathcal{N}(\pmb{\theta}_{MAP}, \mathbf{H}^{-1}).$$ 
Currently, Laplace approximation can be easily implemented in NN with the library designed by Immer et. al [^9]. However, the main bottleneck of this technique comes from computation and memory.


## Contribution
Here, we provide some notebooks in which we compare these approaches leveraging two Regression datasets: a Simulated dataset and Boston dataset. The Simulated dataset was designed to describe the meaning of both aleatoric and epistemic uncertainty and, hence, how the BDL techniques should estimate them. We compare them in terms of accuracy (MAE), calibration (AUSE) and likelihood (NLL). 
![.](/Images/SimulatedRegDataset.PNG)

## Results
Eventually, the performance of previous techniques in both datasets is, 
![.](/Images/BDLResults.PNG)
In fact, this performance may be visualized in the next plots.
### Simulated Dataset
![.](/Images/BDLIntro.PNG)
### Boston Dataset
![.](/Images/BostonBDLTechniques.PNG)

## Bibliography
[^1]: Yiwoong Choi, Dayoung Chun, Hyun Kim, and Hyuk-Jae Lee. Gaussian yolov3: An accurate and fast object detector using localization uncertainty for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 502–511, 2019.
[^2]: Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mohammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra Acharya, et al. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information Fusion, 76:243–297, 2021.
[^3]: https://brendanhasz.github.io/2019/07/23/bayesian-density-net.html
[^4]: Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.
[^5]: Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016.
[^6]: David JC MacKay. Bayesian interpolation. Neural computation, 4(3):415–447, 1992.
[^7]: Andrew YK Foong, Yingzhen Li, José Miguel Hernández-Lobato, and Richard E Turner. ’in-between’uncertainty in bayesian neural networks. arXiv preprint arXiv:1906.11537, 2019.
[^8]: Javier Antorán, David Janz, James U Allingham, Erik Daxberger, Riccardo Rb Barbano, Eric Nalisnick, and José Miguel Hernández-Lobato. Adapting the linearised laplace model evidence for modern deep learning. In International Conference on Machine Learning, pages 796–821. PMLR, 2022
[^9]:  https://github.com/AlexImmer/Laplace.

