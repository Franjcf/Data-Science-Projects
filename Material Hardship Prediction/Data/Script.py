######### loading all the libraries

import pandas as pd
import numpy as np

####Importing Classifier Libraries

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
#rom sklearn.linear_model import Ridge
#rom sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier

####Importing Performance Metrics

from sklearn.metrics import roc_curve, auc, roc_auc_score,average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#####Importing Feature Selection algorithms

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold


####### First we do some pre-proccesing in order to make sure we have a complete data set (elimante NA and bad values)

def fillMissingData(inputcsv, outputcsv):
    
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    
    ## replace NA's with mean mode or median (see how perfermance changces)
    #df = df.fillna(df.mode().iloc[0]) #not all of them have modes
    #df = df.fillna(df.mean())
    #df = df.fillna(df.median())

    # if still NA, replace with a value
    #df = df.fillna(value=-1)
    
    # replace negative values with a value
    #num[num < 0] = -1
   
    #If neccesary, turning al negative and "other" values into Nana
    #df[num < 0] = np.NaN
    #df.replace(["NaN", 'NaT',"Missing","Other"], np.nan, inplace = True)
    
    #If neccesary, replacing all negative values with the median
    #df = df.fillna(df.median())
    
    #If neccesary, converting everything into answered (True) and non-answered (False)
    num = df._get_numeric_data()
    num[num.iloc[:,1:] >= 0] = 1
    num[num.iloc[:,1:] < 0] = 0
    df = df.fillna(value=0)
    df = num
    
     #droppingAllcolumns that are still Na
    df.dropna(axis='columns',inplace = True)
    
    # write filled outputcsv
    df.to_csv(outputcsv, index=False)
    
# Pre-proccesing the features:
fillMissingData('background.csv', 'YesNoAnswered.csv')


#######Now We choose the type of data that we want

#featureData = pd.read_csv('dataOutputBig.csv', low_memory=False) #with negative values
#featureData = pd.read_csv('dataOutputSmall.csv', low_memory=False) #removing all Na and neg values
#featureData = pd.read_csv('dataOutputAllMedian.csv', low_memory=False) #with negative values and NaN as the mean an
featureData = pd.read_csv('YesNoAnswered.csv', low_memory=False) #Pitting negative and Nan as False and all else as True

#print(featureData.shape)
#print(featureData)


## Separating the data and getting only the numerical values
numericData = featureData.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

#print(numericData.shape)
#pint(featureData)

##########Separating Boolean Data and Continous Data

bool_cols = [col for col in numericData 
             if numericData[[col]].isin([0, 1]).all().values]

continiousData = numericData.drop(bool_cols, axis=1)
bool_cols=['challengeID'] + bool_cols
binaryData = numericData[bool_cols]

#print(numericData.shape)
#print(continiousData.shape)
#print(binaryData.shape)


########Pre-proccesing Outcomes

def fillMissingOutcomes(inputcsv, outputcsv):
    
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
   
    # replace NA's with mode or median or mean
    #df = df.fillna(df.mode().iloc[0])
    #df = df.fillna(df.median())
    #df = df.fillna(df.mean())
    
    df.dropna(axis='rows',inplace = True)
    
    # write filled outputcsv
    df.to_csv(outputcsv, index=False)




########Pre-proccesing the training results 

fillMissingOutcomes('train.csv', 'trainOutput.csv')
trainOut = pd.read_csv('trainOutput.csv', low_memory=False)
#print(train)

#########Pre-proccesing the test results 

fillMissingOutcomes('test.csv', 'testOutput.csv')
testOut = pd.read_csv('testOutput.csv', low_memory=False)
#print(trainOut.shape)
#print(testOut.shape)

######### choose the data set to use:

#data = featureData
data = numericData
#data = binaryData
#data = continiousData

#print(data.shape)

########do a variance test to remove features that dont have any variance 

sel = VarianceThreshold(threshold=0.1) #we are not looking for eceptions here 
sel.fit(data)
data = data[data.columns[sel.get_support(indices=True)]]
#print(data.shape)

###########Need to separate the initial data into the training and testings sets

trainIndex = (list(set(trainOut['challengeID']).intersection(data['challengeID'])))
trainIndex.sort()
trainIndex2 = [x - 1 for x in trainIndex] #making sure we get the correct indexes in list to math the ID
testIndex = (list(set(testOut['challengeID']).intersection(data['challengeID'])))
testIndex.sort()
testIndex2 = [x - 1 for x in testIndex]

train_data = data.iloc[trainIndex2]
test_data  = data.iloc[testIndex2]

#########reindexing the data

test_data=test_data.reset_index(drop=True)
train_data=train_data.reset_index(drop=True)

########Making sure all the output boolean data is in boolean type

testOut["layoff"]=testOut["layoff"].astype('bool')
testOut["eviction"]=testOut["eviction"].astype('bool')
testOut["jobTraining"]=testOut["jobTraining"].astype('bool')
trainOut["layoff"]=trainOut["layoff"].astype('bool')
trainOut["eviction"]=trainOut["eviction"].astype('bool')
trainOut["jobTraining"]=trainOut["jobTraining"].astype('bool')

#print(train_data.shape)
#print(trainOut.shape)

#########Making Continuous Outcomes into binary data

ContiniousOutcome = 'materialHardship'

gpaAvg = sum(trainOut[ContiniousOutcome])/len((trainOut[ContiniousOutcome]))

for n in list(range(0,len(trainOut[ContiniousOutcome]))):
    if trainOut[ContiniousOutcome].iloc[n] < gpaAvg:
        trainOut[ContiniousOutcome].iloc[n] = False
    else:
        trainOut[ContiniousOutcome].iloc[n] = True
        
print(trainOut[ContiniousOutcome])

for m in list(range(0,len(testOut[ContiniousOutcome]))):
    if testOut[ContiniousOutcome].iloc[m] < gpaAvg:
        testOut[ContiniousOutcome].iloc[m] = False
    else:
        testOut[ContiniousOutcome].iloc[m] = True
        
#print(testOut[ContiniousOutcome])


#######Defining a vector with all the features we want to test

features = list(test_data)
features = features[1:] #removing the Challenge ID from collumns


######## Choosing best hyper Parameters from cross validation of Logistic Regression
########(THIS TAKES A WHILE SO ITS COMMENTED OUT)

c=[0.001,0.01,0.1,1,10] ##hyper parameter vector

#estimatorLog = LogisticRegression(penalty = "l1",solver='saga',max_iter=1000)
#parametersLog = {"C":c}
#CV = GridSearchCV(estimatorLog, parametersLog, cv=5, refit=True, scoring='precision')
#
#CV.fit(train_data[features].values,trainOut['jobTraining'].values.ravel())
#print(CV.best_estimator_)
#print('score',CV.best_score_)
#print(CV.scorer_)
#
#bestLog = CV.best_estimator_
#LogCoefficients = bestLog.fit(train_data[features].values, trainOut['jobTraining'].values.ravel()).coef_
#LogPredicted = bestLog.predict(test_data[features].values)
#LogScore=bestLog.score(test_data[features].values,testOut['jobTraining'].values.ravel())
#print('accuracy:',LogScore)
#
##print
#plt.figure()
#plt.scatter(test_data[features].iloc[:,0],LogPredicted)
#plt.scatter(test_data[features].iloc[:,0],testOut['layoff'])
#plt.title("Fitted Model Real vs Predicted Values")
#plt.xlabel(features[0])
#plt.ylabel('layoff')
#plt.legend(["Predicted","Real Values"])


##########using Logistic Regression to Obtain the best "N" Parameters based on CV results

outcome = "jobTraining" #########define outcome we want to fit data to

log = LogisticRegression(penalty = "l1",solver='liblinear',max_iter=1000,C=0.1)
log.fit(train_data[features], trainOut[outcome])
predictedLog = log.predict(test_data[features])
probs = log.predict_proba(test_data[features])
probs = probs[:, 1]
scoreLog = log.score(test_data[features],testOut[outcome])
print("Accuracy:","\n",scoreLog)
print("\n classification report: \n",classification_report(testOut[outcome],predictedLog))
print("ROC area under curve:", roc_auc_score(testOut[outcome],probs))
print("Confusion Matrix:","\n",confusion_matrix(testOut[outcome],predictedLog),"\n")
print("Precision Score:",average_precision_score(testOut[outcome],predictedLog),"\n\n")

coeffLog = log.coef_

#print(np.sum(predictedLog))
#print(np.sum(testOut[outcome].values.ravel()))
#print(predictedLog.shape)

#####Getting the best 20 features

LogGoodFeatures=[]
LogGoodCoeff=[]

for i in range(len(features)):
    if abs(coeffLog[0,i])>0.0545: #####Change this number to get desired number of features
        LogGoodFeatures.append(features[i])
        LogGoodCoeff.append(coeffLog[0,i])
        
#print(len(LogGoodFeatures))

plt.figure()
plt.plot(range(len(LogGoodFeatures)), LogGoodCoeff)
plt.xticks(range(len(LogGoodFeatures)), LogGoodFeatures, rotation=90) 
plt.xlabel ('Features')
plt.ylabel ('Coefficients')

n=0
#######writing a text file with most representative words
with open('bestFeaturesLogYesNoJobTraining.txt', 'w') as f:
    for item in LogGoodFeatures:
        f.write("%s\n" % LogGoodFeatures[n])
        n = n + 1


########predicting performance based on best logistic regression features

log = LogisticRegression(solver='liblinear',multi_class='ovr')
log.fit(train_data[LogGoodFeatures], trainOut[outcome])
predictedLog = log.predict(test_data[LogGoodFeatures])
probs = log.predict_proba(test_data[LogGoodFeatures])
probs = probs[:, 1]
scoreLog = log.score(test_data[LogGoodFeatures],testOut[outcome])
print("Accuracy:","\n",scoreLog)
print("\n classification report: \n",classification_report(testOut[outcome],predictedLog))
print("ROC area under curve:", roc_auc_score(testOut[outcome],probs))
print("Confusion Matrix:","\n",confusion_matrix(testOut[outcome],predictedLog),"\n")
print("Precision Score:",average_precision_score(testOut[outcome],predictedLog),"\n\n")

############ Feature Selection based on Recursive Feature Elimination
###########(THIS ALSO TAKES A WHILE SO ITS COMMENTED OUT)

#numFeatures = 20 ########Define number of features that we want to use reduce too. 
#
#rfe = RFE(LogisticRegression(solver='liblinear',multi_class='ovr'),numFeatures) 
#rfe = rfe.fit(train_data[features], trainOut['materialHardship'])
#
#bestFeatures = []
#num = 0 
#
##choosing the best words from the rfe analysis and putting them into a dictinoary
#for bool in rfe.support_:
#	if bool  == True:
#		bestFeatures.append(features[num]) 
#	num =num+1
#
#print(bestFeatures) 
#
#n=0
##writing a text file with most representative words
#with open('bestFeaturesRFEYesNo.txt', 'w') as f:
#    for item in bestFeatures:
#        f.write("%s\n" % bestFeatures[n])
#        n = n + 1

########calculating performance with best "n" RFE features

bestFeaturesRFE = pd.read_csv('bestFeaturesRFEYesNo.txt', low_memory=False,header=None)
bestFeaturesRFE = list(bestFeaturesRFE.iloc[:,0])
 
log = LogisticRegression(solver='liblinear',multi_class='ovr')
log.fit(train_data[bestFeaturesRFE], trainOut[outcome])
predictedLog = log.predict(test_data[bestFeaturesRFE])
probs = log.predict_proba(test_data[bestFeaturesRFE])
probs = probs[:, 1]
scoreLog = log.score(test_data[bestFeaturesRFE],testOut[outcome])
coeffRFE = log.coef_
print("Accuracy:","\n",scoreLog)
print("\n classification report: \n",classification_report(testOut[outcome],predictedLog))
print("ROC area under curve:", roc_auc_score(testOut[outcome],probs))
print("Confusion Matrix:","\n",confusion_matrix(testOut[outcome],predictedLog),"\n")
print("Precision Score:",average_precision_score(testOut[outcome],predictedLog),"\n\n")

plt.figure()
plt.plot(range(len(bestFeaturesRFE)), coeffRFE.T)
plt.xticks(range(len(bestFeaturesRFE)), bestFeaturesRFE, rotation=90) 


#############Calculating Overall Fit with Random Forest Classifier (can also change to Naive bases).
 
from sklearn.naive_bayes import MultinomialNB

outcome = "materialHardship" ##########Choose outcome to fit

log = RandomForestClassifier(random_state=0)
#log = MultinomialNB()
log.fit(train_data[features], trainOut[outcome])
predictedLog2 = log.predict(test_data[features])
scoreLog2 = log.score(test_data[features],testOut[outcome])
print("Accuracy:","\n",scoreLog2)
print("\n classification report: \n",classification_report(testOut[outcome],predictedLog2))
print("Confusion Matrix:","\n",confusion_matrix(testOut[outcome],predictedLog2),"\n")
print("Precision Score:",average_precision_score(testOut[outcome],predictedLog2),"\n\n")

importancesLog2 = log.feature_importances_

print(np.sum(predictedLog2))
print(np.sum(testOut[outcome].values.ravel()))
print(importancesLog2)

############## Getting the best "N" features from Random forest

RF = []
RFCoeff=[]

for i in range(len(features)):
    if abs(importancesLog2[i])>0.0029: ##Change this number to get desired # of features
        RF.append(features[i])
        RFCoeff.append(importancesLog2[i])
        
print(len(RF))

plt.figure()
plt.plot(range(len(RF)), RFCoeff)
plt.xticks(range(len(RF)), RF, rotation=90) 


###########writing a text file with most representative words

n=0
with open('bestFeaturesRFMaterialHardship.txt', 'w') as f:
    for item in RF:
        f.write("%s\n" % RF[n])
        n = n + 1

############ Calculating Performance with N best features of Random Forest

log.fit(train_data[RF], trainOut[outcome])
predictedLog2 = log.predict(test_data[RF])
scoreLog2 = log.score(test_data[RF],testOut[outcome])
print("Accuracy:","\n",scoreLog2)
print("\n classification report: \n",classification_report(testOut[outcome],predictedLog2))
print("Confusion Matrix:","\n",confusion_matrix(testOut[outcome],predictedLog2),"\n")
print("Precision Score:",average_precision_score(testOut[outcome],predictedLog2),"\n\n")
