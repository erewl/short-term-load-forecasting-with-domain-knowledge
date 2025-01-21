# Incorporating Domain Knowledge for Short-Term Electricity Load Forecasting


## About

This repository is part of my Master thesis __"Incorporating Domain Knowledge for Short-Term Electricity Load Forecasting"__ where I explored the application of a custom loss function to a short-term load forecasting problem.
The loss function is designed and inspired by the concept of _semantic loss_.

Semantic loss is a technique to encode background knowledge into a neural network, specifically, into the training process of said neural network, and is part of the larger research area of _Neuro-Symbolic AI_. 

For this project, I specially looked into transforming the loss function based on existing knowledge from the energy domain, and focussed on three business rules that are integrated into the loss function:
1. a consumer cluster is never able to produce any electricity
1. a prosumer cluster is never able to produce any electricity at night
1. morning and evening peaks are time spans with high imbalance costs, and the errors during those times should be kept low

Their mathematical notation can be found in _Appendix B_ of the attached thesis document and their respective python implementation is in the `darts.loss` package of this repository.


## Installation instructions
It is packaged with _poetry_ and runs on Python 3.11

Install the dependencies (from root directory) with: 
```bash
poetry install 
# or
poetry install --directory=/path/to/venv # if you want to specify the venv directory
```
That will create a virtual environment with all dependencies listed in `pyproject.toml`


## Scripts

### Data Cleaning and Aggregation
Takes in a dataframe with multiple smart meter time series and cleans them the following:

- filter out time series with missing values
- per time series: calculate the difference between each time step to get usage data from meter readings
- aggregate all time series into one by summing them by time step
- detect and interpolate outliers via interquartile range

### Training

With this script, the Temporal Fusion Transformer is trained.


### Hyperparameter Tuning
With `main_finetune.py` the hyper parameter tuning session is executed. Similar to the training script 

### Predicting


## Folder structure
