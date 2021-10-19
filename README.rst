================================================================================
Data Science and Machine Learning Projects
================================================================================

This directory contains a representative sample of many of the data science and machine learning projects that I have worked on throughout the years. These include both work-related projects as well as personal projects. Please refer to table of contents (and the linked sub-directory) for a quick explanation of each project and the relevant skills used. 

.. contents::

Note that the code presented here is formatted as a jupyter notebook using Python 3.8, scikit-learn 0.24.2, and TensorFlow 2.4.1.
If you want to run this code on your own please check that you are running the correct versions.

This repository was created by Francisco J. Carrillo

----------------------------------------------------------------------------

################################################################################
Prediction of Stochastic Clogging Processes
################################################################################

**TLDR**:  In this project, I used machine learning, computational fluid dynamics, and discrete element methods to predict stochastic clogging processes. The resulting Extremely Randomized Trees algorithm is able to predict clogging in unseen systems with a classification accuracy of 0.96.

**Description**: The accumulation of discrete masses within constrained flow conduits is an common phenomenon within both natural and industrial settings: it describes the clogging of pipes, roads, oil reservoirs, rivers, and arteries. In this project I used Computational Fluid Mechanics and Discrete Element Models to run over 2,000 different clogging simulations in randomly-generated porous media in order to train and evaluate the clogging prediction performance of several Machine Learning algorithms. The enclosed jupyter notebook contains the python scripts that I wrote to train and evaluate said algorithms. The resulting best-performing classifier (an Extremely Randomized Trees algorithm) is able to predict clogging a-priori with an accuracy of 0.96 and 0.91 in numerical and experimental systems, respectively. Similarly, the best performing regressor (also a decision tree-based algorithm) is able to achieve an R^2 value of 0.93 when predicting the degree of clogging in said systems. I believe this standardized computational tool has the potential to help evaluate the design process of engineered and natural porous media. 

If you are interested, you can find the complete academic manuscript and trained joblib models within the associated directory.

**Skills**: Python, C++, Scikit-learn, XGBoost, Pandas, Numpy, Neural Networks, Decision Tree Clasifiers, Classification, Regression. 

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
English Language Translation with Transformers (Neural Networks + Attention)
################################################################################

**TLDR**: I programmed a transformer to translate English to Portuguese.

**Description**: In this project, I implemented the transformer outlined in the landmark paper "Attention is All You Need" by Vaswani et. al., 2017 into TensorFlow for language translation from English to Portuguese. I programmed this particular model to show how the algorithm gradually learns and "understands" the languages in real time during training. I also made it possible to configure the transformer with differing amounts of encoder and decoder layers, making it a more flexible configuration. 

**Skills**: TensorFlow, OPP, Speech Synthesis, Context Analysis, Deep Learning, Natural Language Processing, Sequence-to-Sequence Models

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/language_translation_transformers/Transformer.ipynb

.. figure:: /images/positional_encoding.png
    :align: right
    :alt: alternate text
    :figclass: align-right

----------------------------------------------------------------------------
    
################################################################################
Ethereum Price Prediction Using LSTMs
################################################################################

**TLDR**: I used LSTM Recurrent Neural Networks to predict the price of Ethereum as a function of previous prices and Reddit comments. I conclude that creating an algorithm that will make you a billionaire is harder than it looks. 

**Description**: Ethereum is a blockchain-based network used for the creation and execution of "smart contracts". These special contracts can be used for a wide range of applications:  from confirming basic cryptocurrency transactions (lending, payments, ext...), the creation on "Non-Fungible Tokens", to the implementation of Decentralized Finance networks. Unfortunately, this network is mostly known for the ample speculation related to its native cryptocurrency "Ether". This has led to wide swings in its price and high volatility. In this project I use Long short-term memory (LSTM) Recurrent Neural Networks (RNN) to try and predict the price of Ether as a function of previous prices and sentiment analysis based on crowd sentiment on "Reddit" (a social media platform). The use of LSTM's allows for the neural network to maintain a "memory" of relevant past events in an effort to increase prediction accuracy. 

**Skills**: TensorFlow, Web scraping API calls, OOP, Reccurent Neural Networks, LSTM, Time-series prediction 

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/Ethereum_price_prediction/ETH_prediction.ipynb

.. figure:: /images/ETH_prediction.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

################################################################################
Fine-Tuning BERT for Sentiment Analysis
################################################################################

**TLDR**: I fine-tuned the BERT transformer architecture to be able to identify positive and negative sentiment in online product reviews. The result is a classifier that can accurately infer the intent of each reviewer. 

**Description**: BERT (Bidirectional Encoder Representations from Transformers) is a novel neural network architecture developed by Google to capture the inter- and intra-sentence context within a text corpus. In this project, I built a classifier on top of said architecture in order to fine-tune the encoder weights within BERT and infer the encoded sentiment within online product reviews. 

**Skills**: TensorFlow, NLP, Transformers, Deep Learning, Sentiment Analysis, Context Capturing

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/sentiment_analysis_BERT/sentiment_analysis_BERT.ipynb

.. figure:: /images/bert_git.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

##################################################
Latent Feature Analysis of "OkCupid" User Profiles
##################################################

**TLDR**: I used LDA to find the optimal number of interpretable latent categories that can accurately classify 60,000 OkCupid users. I also found that drug-loving-atheists are the least likely type of person to find a suitable match. 

**Description**: The study and identification of the hidden (i.e. latent) features on data sets has far-reaching implications in the fields of data science. Potential (and current) applications of latent feature analysis includes the development of search engines, the creation of stock trading algorithms, population analysis, and the sorting of people into groups (for commercial, dating, and/or policy purposes). In this project I used ``One-Hot" encoding, natural language processing (``Bag of Words"), and Latent Dirichlet Allocation to process and analyze the data from 59946 real OkCupid dating profiles originating from the San Francisco Bay Area. In particular I studied the relationship between the prevalence of ``Tweeners'' (users who are sorted into several groups) and the hyper parameter ``k'' (total number of groups). We conclude that, for this data set, ``k" has an optimal value of 7, which dramatically decreases the number of tweeners while still being an interpretable and manageable number of groups. The resulting analysis sorted users into groups composed of 1) intellectuals 2) educated white people 3) artistic people, 4) active people, 5) hipsters, 6) people who love life,  and 7) social people. Furthermore, we identified that tweeners tend to be part of a group which consists of "drug-loving atheists". 

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

**TLDR**:  In this project I optimized the number of features needed to accurately classify Amazon customer reviews into positive reviews and negative reviews. The marginal increase in accuracy becomes negligible after sampling 202 different words. 

**Description**: The application of data science to sentiment analysis has become essential in the development of successful online products, be it in the areas of marketing (Google), entertainment (YouTube), retail (Amazon), and communication (Microsoft). Data science has allowed these sectors to monitor and influence consumer behavior, effectively changing the way that companies interact with their consumers. Direct contact is no longer strictly necessary, it is sufficient to analyze comments, web searches, messages, or product reviews to obtain the consumers’ reaction to a new product or a change in services. In this project, I present an analysis of five different classifiers on a data set comprising of 3000 online reviews labeled as either ”positive” and ”negative”. I compare and contrast the classifiers’ ability to correctly predict a review label based on a ”bag of words” representation and by taking into account the length of said reviews. Furthermore I studied the effects of feature selection (number of words sampled) on classifier performance. I conclude that the Logistic Regression classifier works best when compared to its counterparts, as it requires the least amount of features while obtaining the best performance in 4 out of 6 metrics. Finally, I conclude that review length is not a good predictor of sentiment.

If you are interested, you can find a complete report of all the findings within the associated directory.

**Skills**: Natural Language Processing, Pandas, Naive-Bayes, Decision Trees, Logistic Regression, Numpy, Data Processing

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/sentiment_analysis_Amazon_reviews/Sentiment%20Analysis.ipynb

.. figure:: /images/sentiment_analysis_graphs.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

################################################################################
Unconstrained Optimization Using Gradient Decent 
################################################################################

**TLDR**: Gradient decent optimization works best when using variable step sizes that are dictated by the eigenvalues of Hessian data matrix.

**Description**: In this mini-project I code gradient decent from scratch to solve linear regression and ridge regularization problems. These particular problems were chosen because their analytical solutions are well-known. Furthermore I investigate how the gradient step size affects the rate of convergence of the underlying optimization problem. I then procced by calculating the largest and smallest eigenvalues of the second derivative of objective function in order to set optimal step size and to find the lower bound the rate of convergence. Finally I investigate how the regularization term "lambda" affects said convergence rate. 

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

**TLDR**: Principal Component Analysis can help improve the performance of face-recognition algorithms by filtering-out random noise and unnecessary image features. 

**Description**: Face recognition is an ubiquitous feature in today's technological landscape: it is used within our phones, photo applications, internet communications, and even self-maneuvering machines. However, images are notoriously information-heavy, leading to slow algorithms and large data repositories. In this mini-project, I investigate the application of Singular Value Decomposition and Principal Component Analysis into the area of facial recognition. The results are fairly intuitive: the accuracy of a face recognition algorithm increases as we increase the number of principal components we use to represent a given image. However, this correlation is non-monotonic, leading to quick diminishing returns in accuracy as we get to use around principal 100 components. Therefore, it abundantly clear that we can use PCA to optimize our face-classification algorithms. 

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

**TLDR**: 4th order inhomogeneous kernels and gaussian RBF kernels can help improve the labelling accuracy of handwritten digits and the detection of human liver disorders. 

**Description**: The use of kernels have revolutionized the way we analyze data. They allow us to effectively project raw data into previously-unavailable dimensional spaces in order to produce more-easily classifiable data. The best part is that they can be readily implemented within most Machine Learning algorithms and are not computationally prohibitive in most cases. In this project, I implement (from scratch) several kernels methods into Principal Component Analysis, Support Vector Machines, and K-Nearest Neighbors algorithms. These kernels include a 3rd order inhomogeneous kernel, 4th order inhomogeneous kernel, and gaussian RBF kernels. The ultimate goal is to improve the detection accuracy of said algorithms in the detection of handwritten digits and human liver disorders. 

**Skills**: Kernels Methods, Image Recognition, Data Analysis, Linear Algebra, Support Vector Machines, Principal Component Analysis, KNN.

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/kernel_PCA_SVD_KNN/kernel_PCA_SVD_KNN.ipynb

.. figure:: /images/kernel_PCA.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------

################################################################################
Feature Analysis of Trending YouTube Videos Across the USA, Great Britain, Canada and Mexico
################################################################################

**TLDR**: In this project I used LDA and several type of classifiers in order to identify the features that can predict if a YouTube video will go into the trending page of a particular country. These features varied across different countries.

**Description**: YouTube is the most popular video streaming platform on the planet. With about 2 billion monthly users spread across over 100 countries, it is fair to say that the themes and topics present in the ”trending” videos of this platform are a good reflection of social trends at the national and global level. In this project we used latent feature analysis on data pertaining to YouTube’s trending videos in order to identify topics that describe the sociocultural similarities and differences between various countries. Furthermore, through the use of different classifier models, we identified which features are good predictors of a video’s future ‘trendability” across different nationalities. Furthermore, part of this effort was to identify if having fully capitalized words (i.e. NEW, OFFICIAL, ext...) in a video’s description is a good indication of future trendability. We concluded that it is not. Finally I summarize all these results by creating a hypothetical ‘perfect’ trending video for each country such as to highlight the uniqueness of each nationality.

If you are interested, you can find a complete report of all the findings within the associated directory.

**Skills**:  Data Processing, Pattern Recognition, Language Processing, Latent Dirichlet Allocation, Python, Pandas

**Link to Notebook**: https://github.com/Franjcf/Data-Science-Projects/blob/main/YouTube_video_trending_analysis/trending_YouTube_videos_analysis.ipynb

.. figure:: /images/trending_videos.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
----------------------------------------------------------------------------
