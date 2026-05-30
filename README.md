# Machine Learning from First Principles

**Derive it. Build it. Understand it.**

A collection of educational Jupyter notebooks that derive, implement, and apply core machine learning algorithms **from scratch**, with full mathematical proofs, validated against standard libraries on real-world datasets.

## Learning Path

```
1.  Linear Regression              ──► Foundation: cost functions, gradient descent, regularization
        │
2.  Bayesian Linear Regression     ──► Bayesian: posterior over weights, predictive uncertainty, marginal likelihood
        │
3.  Logistic Regression            ──► Extends linear to classification: sigmoid, cross-entropy, MLE
        │
4.  K-Nearest Neighbors            ──► Non-parametric: distance metrics, curse of dimensionality
        │
5.  Naive Bayes                    ──► Probabilistic: Bayes' theorem, conditional independence
        │
6.  Decision Trees                 ──► Non-linear: entropy, information gain, pruning
        │
7.  Support Vector Machines        ──► Optimization: max margin, Lagrangian duality, kernel trick
        │
8.  Neural Networks (MLP)          ──► Deep learning: backpropagation, activations, optimizers
        │
9.  Bayesian Networks              ──► Graphical models: DAGs, inference, d-separation
        │
10. Gaussian Processes             ──► Bayesian non-parametric: kernels, posterior, uncertainty
        │
11. Hidden Markov Models           ──► Sequential: Forward, Viterbi, Baum-Welch algorithms
```

## Notebooks

| # | Algorithm | Problem | Dataset | Key Math | Framework |
|---|-----------|---------|---------|----------|-----------|
| 1 | **Linear Regression** | Building energy efficiency | UCI Energy Efficiency | Normal Equation, GD, Ridge/Lasso | NumPy + sklearn |
| 2 | **Bayesian Linear Regression** | Building energy efficiency | UCI Energy Efficiency | Posterior, Predictive Uncertainty, Marginal Likelihood | NumPy + sklearn |
| 3 | **Logistic Regression** | Heart disease prediction | UCI Heart Disease | Sigmoid, MLE, Cross-Entropy, Newton | NumPy + sklearn |
| 4 | **K-Nearest Neighbors** | Wine quality classification | UCI Wine Quality | Distance metrics, Curse of Dimensionality | NumPy + sklearn |
| 5 | **Naive Bayes** | SMS spam detection | UCI SMS Spam | Bayes' Theorem, MAP, Laplace Smoothing | NumPy + sklearn |
| 6 | **Decision Trees** | Predictive maintenance | Turbofan Sensor Data | Entropy, Gini, CART, Pruning | NumPy + sklearn |
| 7 | **Support Vector Machines** | Breast cancer diagnosis | Wisconsin WDBC | Lagrangian Dual, KKT, Kernel Trick | NumPy + sklearn |
| 8 | **Neural Networks (MLP)** | Fashion image classification | Fashion-MNIST | Backpropagation, Adam, Xavier Init | NumPy + PyTorch |
| 9 | **Bayesian Networks** | Heart disease diagnosis | Cleveland Heart Disease | DAGs, d-Separation, Variable Elimination, Gibbs | NumPy + pgmpy |
| 10 | **Gaussian Processes** | Air quality prediction | Beijing PM2.5 | GP Posterior, Kernels, Marginal Likelihood, Weight-Space View | NumPy + GPyTorch |
| 11 | **Hidden Markov Models** | Part-of-speech tagging | Brown Corpus | Forward, Viterbi, Baum-Welch | NumPy + hmmlearn |

## Prerequisites

- **Linear algebra**: matrix multiplication, transposes, inverses, eigenvalues
- **Calculus**: partial derivatives, chain rule, gradients
- **Probability**: conditional probability, Bayes' theorem, distributions
- **Python**: NumPy, Matplotlib, basic OOP

## Structure

Each notebook folder contains:
```
NN_Algorithm_Name/
  NN_Algorithm_Name.ipynb   # Main notebook (40–55 cells)
  requirements.txt          # Minimal dependencies
  illustrations/            # Pre-rendered conceptual figures (PNG)
```

## Philosophy

1. **Math first**: every algorithm is derived from first principles before any code is written
2. **From scratch**: core implementations use only NumPy, no black-box library calls
3. **Then validate**: scikit-learn / library implementations verify correctness (±1–2% tolerance)
4. **Real problems**: each notebook solves a unique real-world problem, not toy examples
5. **Reproducible**: seeds set for all random sources; single `requirements.txt` per notebook

## Quick Start

```bash
cd 01_Linear_Regression/
pip install -r requirements.txt
jupyter notebook 01_Linear_Regression.ipynb
```

## License

MIT

## Author

Antonis Kastellakis


