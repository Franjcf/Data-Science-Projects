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

**Skills**: Python, Scikit-learn, Pandas, Numpy, Neural Networks, Decision Tree Clasifiers, Classification, Regression. 

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/clogging_prediction_analysis/clogging_analysis.ipynb

.. figure:: /images/clogging_graph_classification.png
    :align: right
    :alt: alternate text
    :figclass: align-right


