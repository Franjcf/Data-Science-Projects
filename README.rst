================================================================================
Data Science and Machine Learning Projects
================================================================================

This directory contains a representative sample of many of the data science and machine learning projects that I have worked on throughout the years. These include both work-related projects as well as personal projects. Please refer to table of contents (and the linked sub-directory) for a quick explanation of each project and the relevant skills used. 

.. contents::

Note that the code presented here is formated as a jupyter notebook using Python 3.8, scikit-learn 0.24.2, and TensorFlow 2.4.1.
If you want to run this code on your own please check that you are running the correct versions.

This repository was created by Francisco J. Carrillo

----------------------------------------------------------------------------

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
    
----------------------------------------------------------------------------
    
################################################################################
Ethereum Price Prediction Using LSTMs
################################################################################

**Summary**: Ethereum is a blockchain-based network used for the creation and execution of "smart contracts". These special contracts can be used for a wide range of applications:  from confirming basic cryptocurrency transactions (lending, payments, ext...), the creation on "Non-Fungible Tokens", to the implementation of Decentrilized Finance networks. Unfortunately, this network is mostly known for the ample speculation related to its native cryptocurrency "Ether". This has led to wide swings in its price and high volatility. In this project I use Long short-term memory (LSTM) Recurrent Neural Networks (RNN) to try and predict the price of Ether as a function of previous prices and sentiment analysis based on crowd sentiment on "Reddit" (a social media platform). The use of LSTM's allows for the neural network to maintain a "memory" of relevant past events in an effort to increase prediction accuracy. 

**Skills**: TensorFlow, Web scraping API calls, OOP, Reccurent Neural Networks, LSTM, Time-series prediction 

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/Ethereum_price_prediction/ETH_prediction.ipynb

.. figure:: /images/ETH_prediction.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

##################################################
Latent Feature Analysis of "OkCupid" User Profiles
##################################################

**Summary**: The study and identification of the hidden (i.e. latent) features on data sets has far-reaching implications in the fields of data science. Potential (and current) applications of latent feature analysis includes the development of search engines, the creation of stock trading algorithms, population analysis, and the sorting of people into groups (for commercial, dating, and/or policy purposes). In this project I used ``One-Hot" encoding, natural language processing (``Bag of Words"), and Latent Dirichlet Allocation to process and analyze the data from 59946 real OkCupid dating profiles originating from the San Francisco Bay Area. In particular I studied the relationship between the prevalence of ``Tweeners'' (users who are sorted into several groups) and the hyper parameter ``k'' (total number of groups). We conclude that, for this data set, ``k" has an optimal value of 7, which dramatically decreases the number of tweeners while still being an intepretable and manageable number of groups. The resulting analysis sorted users into groups composed of 1) intellectuals 2) educated white people 3) artistic people, 4) active people, 5) hipsters, 6) people who love life,  and 7) social people. Furthermore, we identified that tweeners tend to be part of a group which consists of "drug-loving atheists". 

If you are interested, you can find a complete report of all the findings within the associated directory.

**Skills**: Latent Dirichlet Allocation, Unsupervised models, NLP, Python, Scikit-learn, Cleaning and Preparation of Data. 

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/OKCupid_LDA_analysis/OKCupid_LDA.ipynb

.. figure:: /images/LDA_histograms.PNG
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------
    
################################################################################
Sentiment Analysis of Amazon Customer Reviews
################################################################################

**Summary**: The application of data science to sentiment analysis has become essential in the development of successful online products, be it in the areas of marketing (Google), entertainment (YouTube), retail (Amazon), and communication (Microsoft). Data science has allowed these sectors to monitor and influence consumer behaviour, effectively changing the way that companies interact with their consumers. Direct contact is no longer strictly necessary, it is sufficient to analyze comments, web searches, messages, or product reviews to obtain the consumers’ reaction to a new product or a change in services. In this project, I present an analysis of five different classifiers on a data set comprising of 3000 online reviews labeled as either ”positive” and ”negative”. I compare and contrast the classifiers’ ability to correctly predict a review label based on a ”bag of words” representation and by taking into account the length of said reviews. Furthermore I studied the effects of feature selection (number of words sampled) on classifier performance. I conclude that the Logistic Regression classifier works best when compared to its counterparts, as it requires the least amount of features while obtaining the best performance in 4 out of 6 metrics. Finally, I conclude that review length is not a good predictor of sentiment.

If you are interested, you can find a complete report of all the findings within the associated directory.

**Skills**: Natural Language Proccesing, Pandas, Naive-Bayes, Decision Trees, Logistic Regression, Numpy, Data Proccesing

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/sentiment_analysis_Amazon_reviews/Sentiment%20Analysis.ipynb

.. figure:: /images/sentiment_analysis_graphs.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

################################################################################
Unconstrained Optimization Using Gradient Descent
################################################################################

**Summary**: In this mini-project I code gradient decesnt from scratch to solve linear regression and ridge regularization problems. These particular problems were chosen becauese their analytical solutions are well-known. Furthermore I investigate how the gradient step size affects the rate of convergence of the underlying optimization problem. I then procced by calculating the largest and smallest eigenvalues of the second derivative of objective function in order to set optimal step size and to find the lower bound the rate of convergence. Finally I investigate how the regularization term "lambda" affects said convergence rate. 

**Skills**: Vector Calculus, Linear Algebra, Optimization, Gradient Decent, Python, Ridge and Linear Regression

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/gradient_decent_optimization_and_implementation/unconstrained_optimization_with_gradient_decent.ipynb

.. figure:: /images/optimal_steps_gradient_decent.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

################################################################################
Face Recognition Through Principal Component Analysis
################################################################################

**Summary**: Face recognition is an ubiquitous feature in today's techonological landscape: it is used within our phones, photo applications, internet communications, and even self-manuvering machines. However, images are notoriously information-heavy, leading to slow algorithms and large data repositories. In this mini-project, I investigate the application of Singular Value Decomposition and Principal Component Analysis into the area of facial recognition. The results are fairly intuitive: the accuracy of a face recognition algorithm increases as we increase the number of principal components we use to represent a given image. However, this correlation is non-monotonic, leading to quick diminishing returns in accuracy as we get to use around principal 100 components. Therefore, it abundantly clear that we can use PCA to optimize our face-classification algorithms. 

**Skills**: Image Recognition, Linear Algebra, Python, Principal Component Analysis, K-Nearest Neighbors.

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/face_recognition_PCA/Face_Recognition.ipynb

.. figure:: /images/face_recognition.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

################################################################################
Application of Kernel Methods to PCA, SVM and KNN for Image and Disease recognition 
################################################################################

**Summary**: The use of kernels have revolutionized the way we analyize data. They allow us to effectivly project raw data into previously-unavailable dimensional spaces in order to produce more-easily classifiable data. The best part is that they can be readily implemented within most Machine Learning algorithms and are not computationally prohibitive in most cases. In this project, I implement (from scratch) several kernels methods into Principcal Component Analyisis, Support Vector Machines, and K-Nearest Neighbors algorithms. These kernels include a 3rd order inhomogenious kernel, 4th order inhomogenious kernel, and gaussian RBF kernels. The ultimate goal is to improve the detection accuracy of said algorithms in the detection of handwritten digets and human liver disorders. 

**Skills**: Kernels Methods, Image Recognition, Data Analysis, Linear Algebra, Support Vector Machines, Principal Component Analysis, KNN.

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/kernel_PCA_SVD_KNN/kernel_PCA_SVD_KNN.ipynb

.. figure:: /images/kernel_PCA.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------
