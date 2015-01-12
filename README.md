# sds-models
A small framework to create and validate different models of judgments, focusing on sequential modeling with Markov chains. Provides easy setup of parameter sweeps, model performance metrics and model comparison. Data are passed as CSV. The included class Experiment allows to set up and run one or a series of experiments (creating a model with certain parameters, validating the model, calculating the performance metrics). 

Currently includes these models:
* GMM - A Gaussian mixture model.
* HMM - A hidden Markov model, using the Nltk implementation and maximum-likelihood parameter estimates.
* GMM-HMM - A hidden Markov model with Gaussian mixture emissions.
