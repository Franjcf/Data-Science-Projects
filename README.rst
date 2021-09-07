================================================================================
Data Science Projects
================================================================================

This directory contains a representative sample of some of the data science and machine learning projects that I have worked on throughout the years. These include both work-related projects as well as personal projects. Please refer to table of contentents (and the linked subdirectory) for a quick explanation of each project and the relevant skills used. 

This repository was created by Francisco J. Carrillo

----------------------------------------------------------------------------

.. contents::

################################################################################
General Information
################################################################################

The code presented here is formated to be displayed as a jupyter notebook using Python 3.8, scikit-learn 0.24.2, and TensorFlow 2.4.1.
If you want to run this code on your own please check that you are running the correct version.

################################################################################
Stochastic Clogging Prediction Analysis
################################################################################

**Summary**: The accumulation of discrete masses within constrained flow conduits is an common phenomenon within both natural and industrial settings: it describes the clogging of pipes, roads, oil reservoirs, rivers, and arteries. In this project I used Computational Fluid Mechanics and Discrete Element Models to run over 2,000 different clogging simulations in randomly-generated porous media in order to train and evaluate the clogging prediction performance of several Machine Learning algorithms. The enclosed jupyter notebook contains the python scripts that I wrote to train and evaluate said algorithms. The resulting best-performing classifier (an Extremely Randomized Trees algorithm) is able to predict clogging a-priori with an accuracy of 0.96 and 0.91 in numerical and experimental systems, respectively. Similarly, the best performing regressor (also a decision tree-based algorithm) is able to achieve an R^2 value of 0.93 when predicting the degree of clogging in said systems. I believe this standardized computational tool has the potential to help evaluate the design process of engineered and natural porous media. 

If you are interested, you can find the complete academic manunscript and trained joblib models within the associated directory.

**Skills**: Python, C++, Scikit-learn, Pandas, Numpy, Neural Networks, Decision Tree Clasifiers, Classification, Regression. 

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/clogging_prediction_analysis/clogging_analysis.ipynb

.. figure:: /images/clogging_graph_classification.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
.. figure:: /images/clogging_extent.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
################################################################################
Ethereum Price Prediction Using LSTMs
################################################################################

**Summary**: Ethereum is a blockchain-based network used for the creation and execution of "smart contracts". These special contracts can be used for a wide range of applications:  from confirming basic cryptocurrency transactions (lending, payments, ext...), the creation on "Non-Fungible Tokens", to the implementation of Decentrilized Finance networks. Unfortunately, this network is mostly known for the ample speculation related to its native cryptocurrency "Ether". This has led to wide swings in its price and high volatility. In this project we use Long short-term memory (LSTM) Recurrent Neural Networks (RNN) to try and predict the price of Ether as a function of previous prices and sentiment analysis based on crowd sentiment on "Reddit" (a social media platform). The use of LSTM's allows for the neural network to maintain a "memory" of relevant past events in an effort to increase prediction accuracy. 

**Skills**: TensorFlow, Web scraping API calls, OOP, Reccurent Neural Networks, LSTM, Time-series prediction 

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/Ethereum_price_prediction/ETH_prediction.ipynb

.. figure:: /images/ETH_prediction.png
    :align: right
    :alt: alternate text
    :figclass: align-right

##################################################
Latent Feature Analysis of "OkCupid" User Profiles
##################################################

**Summary**: The study and identification of the hidden (i.e. latent) features on data sets has far-reaching implications in the fields of data science. Potential (and current) applications of latent feature analysis includes the development of search engines, the creation of stock trading algorithms, population analysis, and the sorting of people into groups (for commercial, dating, and/or policy purposes). In this project I used ``One-Hot" encoding, natural language processing (``Bag of Words"), and Latent Dirichlet Allocation to process and analyze the data from 59946 real OkCupid dating profiles originating from the San Francisco Bay Area. In particular I studied the relationship between the prevalence of ``Tweeners'' (users who are sorted into several groups) and the hyper parameter ``k'' (total number of groups). We conclude that, for this data set, ``k" has an optimal value of 7, which dramatically decreases the number of tweeners while still being an intepretable and manageable number of groups. The resulting analysis sorted users into groups composed of 1) intellectuals 2) educated white people 3) artistic people, 4) active people, 5) hipsters, 6) people who love life,  and 7) social people. Furthermore, we identified that tweeners tend to be part of a group which consists of "drug-loving atheists". 

If you are interested, you can find a complete report of all the findings within the associated directory.

**Skills**: Latent Dirichlet Allocation, Unsupervised models, NLP, Python, Scikit-learn, Cleaning and Preparation of Data. 

**Link to Notebook**: 

.. figure:: /images/ETH_prediction.png
    :align: right
    :alt: alternate text
    :figclass: align-right
