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

**Summary**: The accumulation of discrete masses within constrained flow conduits is an common phenomenon within both natural and industrial settings: it describes the clogging of pipes, roads, oil reservoirs, rivers, and arteries. In this project I used Computational Fluid Mechanics and Discrete Element Models to run over 2,000 different clogging simulations in randomly-generated porous media in order to train and evaluate the clogging prediction performance of several Machine Learning algorithms. The enclosed jupyter notebook contains the python scripts that I wrote to train and evaluate said algorithms. The resulting best-performing classifier (an Extremely Randomized Trees algorithm) is able to predict clogging a-priori with an accuracy of 0.96 and 0.91 in numerical and experimental systems, respectively. Similarly, the best performing regressor (also a decision tree-based algorithm) is able to achieve an $R^2$ value of 0.93 when predicting the degree of clogging in said systems. I believe this standardized computational tool has the potential to help evaluate the design process of engineered and natural porous media. 

If you are interested, you can find the complete academic manunscript and trained models within the associated directory.

**Skills**: Python, Scikit-learn, Pandas, Numpy, Neural Networks, Decision Tree Clasifiers, Classification, Regression. 

**Link to Notebook**: [Clogging Prediction](Data-Science-Projects/Clogging Analysis/Clogging Analysis.ipynb)

.. figure:: /images/fracturing.png
    :align: right
    :alt: alternate text
    :figclass: align-right

################################################################################
Running the Tutorials
################################################################################

To test if the solver was installed correctly, you can run all the included tutorial cases by typing the following code within the "tutorials" subdirectory:

.. code-block:: bash

  python runTutorials.py

Note that this will only run each case for a single time step. Still, it might take a while. Also make sure to use python2 to run the associated script.  

----------------------------------------------------------------------------

Each tutorial directory contains "run" and "clean" files to test installation and validate the solver. To run a particular tutorial for more than a single time step just replace "writeNow" with "endTime" within its "system/controlDict" file. Then you can run said tutorial by typing:

.. code-block:: bash

  ./run

or equivalently, for linear elastic systems:

.. code-block:: bash

  elasticHBIF
  
and for plastic systems:

.. code-block:: bash

  plasticHBIF

To clean the directory:

.. code-block:: bash

  ./clean

################################################################################
List of Included Cases
################################################################################

**Linear Elastic Cases**

- Test cases related to the verification of the solver for poroelastic porous media (Terzaghi consolidation problem and pressure-oscillation in poroelastic core).

.. figure:: /images/poroelastic_oscillation.png 
    :align: right
    :alt: alternate text
    :figclass: align-right

----------------------------------------------------------------------------

**Plastic Cases**

- Test cases related to the verification of the solver for poroplastic porous media (fracturing in a Hele-Shaw cell and in low-permeability formations).

.. figure:: /images/fracturing.png
    :align: right
    :alt: alternate text
    :figclass: align-right

----------------------------------------------------------------------------

**Example Applications/Case Templates**

- Sample cases that show the multi-scale nature of this solver by simulating systems with a combination of porous and free-fluid regions (wave absorption in poroelastic coastal barriers and fracture-driven surface deformation). Each variable within the "0/" directory and the "constant" directory is labeled to make it easier to understand. There is a template for both elastic and plastic systems. 

.. figure:: /images/coastal_barrier.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
.. figure:: /images/surface_fracturing.png
    :align: right
    :alt: alternate text
    :figclass: align-right
    
################################################################################
List of Included Libraries
################################################################################


