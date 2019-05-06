#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bowen Li
"""

#Imports libraries
import pandas as pd
import sklearn.metrics as metrics
#import numpy
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.decomposition import PCA
from prettytable import PrettyTable

def pca(input):
    t = PCA(n_components = 'mle')
    newData = t.fit_transform(input)
    return newData

"""
#Imports data
bestsellersdf = pandas.read_csv('amazonbestsellers.csv', encoding = 'ISO-8859-1')
toysgamesdf = pandas.read_csv('toys.csv', encoding = 'ISO-8859-1')
electronics1df = pandas.read_csv('electronic.csv', encoding = 'ISO-8859-1')
electronics2df = pandas.read_csv('TVsLaptopsDesktopsHTS.csv', encoding = 'ISO-8859-1')
clothesshoesjewelrydf = pandas.read_csv('Clothing, Shoes, Jewelry top 10 categories.csv', encoding = 'ISO-8859-1')
#homekitchendf = pandas.read_csv('.csv', encoding = 'ISO-8859-1')

#Combines data into one data set
amazondf = bestsellersdf.append([toysgamesdf, electronics1df, electronics2df, clothesshoesjewelrydf])
#print(amazondf)
amazondf.to_csv("newamazondat.csv")
"""
#Reads cleaned data and splits it into training and testing data sets
cleanamazondf = pd.read_csv('amazonmodeldata.csv')
cleanamazondf2 = pd.DataFrame(data = cleanamazondf)
features0 = cleanamazondf2[['# of Reviews','Delivery option','Made by','Price','Prime or not','Questions','Stars','Stock']]
targets = cleanamazondf2[['Label']]
features = pca(features0)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.30, random_state = 42)
c, r = y_train.shape
y_train = y_train.values.reshape(c,)
#print(X_test)
#print(y_test)

#####ADD TRUE/FALSE TABLES (CONFUSION TABLE) FOR EACH MODEL

#####ADD GridSearchCV FOR EACH MODEL

#Get confusion table accuracy
def get_accuracy(metrix):
    bs_bs_predicted_accuracy = float(metrix[0][0])/float(metrix[0][0] + metrix[1][0])
    nbs_nbs_predicted_accuracy = float(metrix[1][1])/float(metrix[1][1] + metrix[0][1])
    return nbs_nbs_predicted_accuracy, bs_bs_predicted_accuracy

#Runs KNN algoorithm
knn = KNeighborsClassifier(n_neighbors=3)
knnfitted = knn.fit(X_train, y_train)
knnscores = knnfitted.score(X_test, y_test)
knnpredictions = knnfitted.predict(X_test)
knnconmatrix = metrics.confusion_matrix(y_test, knnpredictions)
print('KNN Confusion Matrix:', '\n', knnconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(knnconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%'  %(get_accuracy(knnconmatrix)[0] * 100))
print('KNN Model Accuracy = %.3f%%' %(knnscores * 100))
params_search = {'n_neighbors': list(range(1, 31))}
knn_paras = GridSearchCV(knn, params_search, n_jobs = -1)
knnfitted_gs = knn_paras.fit(X_train, y_train)
knnscores_gs = knnfitted_gs.score(X_test, y_test)
knnpredictions_gs = knnfitted_gs.predict(X_test)
knnconmatrix_gs = metrics.confusion_matrix(y_test, knnpredictions_gs)
print('KNN GridSearchCV Confusion Matrix:', '\n', knnconmatrix_gs)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(knnconmatrix_gs)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(knnconmatrix_gs)[0] * 100))
print('KNN Model GridSearchCV Accuracy = %.3f%%' %(knnscores_gs * 100))
q_KNN = PrettyTable()
q_KNN.field_names = ['Comparision', 'True positive rate', 'True negative rate', 'Model Accuracy']
q_KNN.add_row(['KNN', 
           '%0.3f%%'%(get_accuracy(knnconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(knnconmatrix)[0] * 100), 
           '%0.3f%%'%(knnscores * 100)])
q_KNN.add_row(['KNN + GridSearch', 
           '%0.3f%%'%(get_accuracy(knnconmatrix_gs)[1] * 100), 
           '%0.3f%%'%(get_accuracy(knnconmatrix_gs)[0] * 100), 
           '%0.3f%%'%(knnscores_gs * 100)])
print(q_KNN, '\n')

#Runs Support Vector Machine algorithm
svmmodel = svm.SVC(C=1, gamma=1)
svmfitted = svmmodel.fit(X_train, y_train)
svmscores = svmfitted.score(X_test, y_test)
svmpredictions = svmfitted.predict(X_test)
svmconmatrix = metrics.confusion_matrix(y_test, svmpredictions)
print('SVM Confusion Matrix:', '\n', svmconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(svmconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(svmconmatrix)[0] * 100))
print('Support Vector Machine Model Accuracy = %.3f%%' %(svmscores * 100))
params_search = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1]}
svmmodel_paras = GridSearchCV(svmmodel, params_search, n_jobs = -1)
svmfitted_gs = svmmodel_paras.fit(X_train, y_train)
svmscores_gs = svmfitted_gs.score(X_test, y_test)
svmpredictions_gs = svmfitted_gs.predict(X_test)
svmconmatrix_gs = metrics.confusion_matrix(y_test, svmpredictions_gs)
print('SVM GridSearchCV Confusion Matrix:', '\n', svmconmatrix_gs)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(svmconmatrix_gs)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(svmconmatrix_gs)[0] * 100))
print('Support Vector Machine Model GridSearchCV Accuracy = %.3f%%' %(svmscores_gs * 100))
q_SVM = PrettyTable()
q_SVM.field_names = ['Comparision', 'True positive rate', 'True negative rate', 'Model Accuracy']
q_SVM.add_row(['SVM', 
           '%0.3f%%'%(get_accuracy(svmconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(svmconmatrix)[0] * 100), 
           '%0.3f%%'%(svmscores * 100)])
q_SVM.add_row(['SVM + GridSearch', 
           '%0.3f%%'%(get_accuracy(svmconmatrix_gs)[1] * 100), 
           '%0.3f%%'%(get_accuracy(svmconmatrix_gs)[0] * 100), 
           '%0.3f%%'%(svmscores_gs * 100)])
print(q_SVM, '\n')

#Runs Random Forest algorithm
rfmodel = RandomForestClassifier(n_estimators = 100)
rffitted = rfmodel.fit(X_train, y_train)
rfscores = rffitted.score(X_test, y_test)
rfpredictions = rffitted.predict(X_test)
rfconmatrix = metrics.confusion_matrix(y_test, rfpredictions)
print('RF Confusion Matrix:', '\n', rfconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(rfconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(rfconmatrix)[0] * 100))
print('Random Forest Model Accuracy = %.3f%%' %(rfscores * 100))
params_search = { 'n_estimators':[10, 100],
                  'criterion':['entropy','gini'],
                  'max_depth':[2, 10, 50, 100],
                  'min_samples_split':[2,5,10],
                  'min_weight_fraction_leaf':[0.0,0.1,0.2,0.3,0.4,0.5]
                  }
rfmodel_paras = GridSearchCV(rfmodel, params_search, n_jobs = -1)
rffitted_gs = rfmodel_paras.fit(X_train, y_train)
rfscores_gs = rffitted_gs.score(X_test, y_test)
rfpredictions_gs = rffitted_gs.predict(X_test)
rfconmatrix_gs = metrics.confusion_matrix(y_test, rfpredictions_gs)
print('Random Forest GridSearchCV Confusion Matrix:', '\n', rfconmatrix_gs)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(rfconmatrix_gs)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(rfconmatrix_gs)[0] * 100))
print('Random Forest Model GridSearchCV Accuracy = %.3f%%' %(rfscores_gs * 100))
q_RF = PrettyTable()
q_RF.field_names = ['Comparision', 'True positive rate', 'True negative rate', 'Model Accuracy']
q_RF.add_row(['RF', 
           '%0.3f%%'%(get_accuracy(rfconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(rfconmatrix)[0] * 100), 
           '%0.3f%%'%(rfscores * 100)])
q_RF.add_row(['RF + GridSearch', 
           '%0.3f%%'%(get_accuracy(rfconmatrix_gs)[1] * 100), 
           '%0.3f%%'%(get_accuracy(rfconmatrix_gs)[0] * 100), 
           '%0.3f%%'%(rfscores_gs * 100)])
print(q_RF, '\n')

#Runs Decision Tree algorithm
dtmodel = DecisionTreeClassifier(random_state = 0)
dtfitted = dtmodel.fit(X_train, y_train)
dtscores = dtfitted.score(X_test, y_test)
dtpredictions = dtfitted.predict(X_test)
dtconmatrix = metrics.confusion_matrix(y_test, dtpredictions)
print('Decision Tree Confusion Matrix:', '\n', dtconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(dtconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(dtconmatrix)[0] * 100))
print('Decision Tree Model Accuracy = %.3f%%' %(dtscores * 100))
params_search = {'max_leaf_nodes': [None, 2, 3, 4, 5 ,6 ,7, 8, 9]}
dtmodel_paras = GridSearchCV(dtmodel, params_search, n_jobs = -1)
dtfitted_gs = dtmodel_paras.fit(X_train, y_train)
dtscores_gs = dtfitted_gs.score(X_test, y_test)
dtpredictions_gs = dtfitted_gs.predict(X_test)
dtconmatrix_gs = metrics.confusion_matrix(y_test, dtpredictions_gs)
print('Decision Tree GridSearchCV Confusion Matrix:', '\n', dtconmatrix_gs)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(dtconmatrix_gs)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(dtconmatrix_gs)[0] * 100))
print('Decision Tree Model GridSearchCV Accuracy = %.3f%%' %(dtscores_gs * 100))
q_DT = PrettyTable()
q_DT.field_names = ['Comparision', 'True positive rate', 'True negative rate', 'Model Accuracy']
q_DT.add_row(['DT', 
           '%0.3f%%'%(get_accuracy(dtconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(dtconmatrix)[0] * 100), 
           '%0.3f%%'%(dtscores * 100)])
q_DT.add_row(['DT + GridSearch', 
           '%0.3f%%'%(get_accuracy(dtconmatrix_gs)[1] * 100), 
           '%0.3f%%'%(get_accuracy(dtconmatrix_gs)[0] * 100), 
           '%0.3f%%'%(dtscores_gs * 100)])
print(q_DT, '\n')

#Runs Logistic Regression algorithm
lrmodel = LogisticRegression()
lrfitted = lrmodel.fit(X_train, y_train)
lrscores = lrfitted.score(X_test, y_test)
lrpredictions = lrfitted.predict(X_test)
lrconmatrix = metrics.confusion_matrix(y_test, lrpredictions)
print('Logistic Regression Confusion Matrix:', '\n', lrconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(lrconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(lrconmatrix)[0] * 100))
print('Logistic Regression Model Accuracy = %.3f%%' %(lrscores * 100))
params_search = {'penalty':['l2'],
                 'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                'solver':['lbfgs'],
                'multi_class':['ovr','multinomial']}
lrmodel_paras = GridSearchCV(lrmodel, params_search, n_jobs = -1)
lrfitted_gs = lrmodel_paras.fit(X_train, y_train)
lrscores_gs = lrfitted_gs.score(X_test, y_test)
lrpredictions_gs = lrfitted_gs.predict(X_test)
lrconmatrix_gs = metrics.confusion_matrix(y_test, lrpredictions_gs)
print('Logistic Regression GridSearchCV Confusion Matrix:', '\n', lrconmatrix_gs)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(lrconmatrix_gs)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(lrconmatrix_gs)[0] * 100))
print('Logistic Regression GridSearchCV Accuracy = %.3f%%' %(lrscores_gs * 100))
q_LR = PrettyTable()
q_LR.field_names = ['Comparision', 'True positive rate', 'True negative rate', 'Model Accuracy']
q_LR.add_row(['LR', 
           '%0.3f%%'%(get_accuracy(lrconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(lrconmatrix)[0] * 100), 
           '%0.3f%%'%(lrscores * 100)])
q_LR.add_row(['LR + GridSearch', 
           '%0.3f%%'%(get_accuracy(lrconmatrix_gs)[1] * 100), 
           '%0.3f%%'%(get_accuracy(lrconmatrix_gs)[0] * 100), 
           '%0.3f%%'%(lrscores_gs * 100)])
print(q_LR, '\n')

#Runs Gaussian Naive Bayes algorithm
nbmodel = GaussianNB()
nbfitted = nbmodel.fit(X_train, y_train)
nbscores = nbfitted.score(X_test, y_test)
nbpredictions = nbfitted.predict(X_test)
nbconmatrix = metrics.confusion_matrix(y_test, nbpredictions)
print('Gaussian NB Confusion Matrix:', '\n', nbconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(nbconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(nbconmatrix)[0] * 100))
print('Gaussian Naive Bayes Model Accuracy = %.3f%%' %(nbscores * 100), '\n')

#Runs Stochastic Gradient Descent algorithm
sgdmodel = SGDClassifier()
sgdfitted = sgdmodel.fit(X_train, y_train)
sgdscores = sgdfitted.score(X_test, y_test)
sgdpredictions = sgdfitted.predict(X_test)
sgdconmatrix = metrics.confusion_matrix(y_test, sgdpredictions)
print('SGD Confusion Matrix:', '\n', sgdconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(sgdconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(sgdconmatrix)[0] * 100))
print('Stochastic Gradient Descent Model Accuracy = %.3f%%' %(sgdscores * 100), '\n')
q_NB_SGD = PrettyTable()
q_NB_SGD.field_names = ['Comparision', 'True positive rate', 'True negative rate', 'Model Accuracy']
q_NB_SGD.add_row(['NB', 
           '%0.3f%%'%(get_accuracy(nbconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(nbconmatrix)[0] * 100), 
           '%0.3f%%'%(nbscores * 100)])
q_NB_SGD.add_row(['SGD', 
           '%0.3f%%'%(get_accuracy(sgdconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(sgdconmatrix)[0] * 100), 
           '%0.3f%%'%(sgdscores * 100)])
print(q_NB_SGD, '\n')

#Runs Gradient Boosting algorithm
gbdtmodel = GradientBoostingClassifier(n_estimators = 60)
gbdtfitted = gbdtmodel.fit(X_train, y_train)
gbdtscores = gbdtfitted.score(X_test, y_test)
gbdtpredictions = gbdtfitted.predict(X_test)
gbdtconmatrix = metrics.confusion_matrix(y_test, gbdtpredictions)
print('GBDT Confusion Matrix:', '\n', gbdtconmatrix)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(gbdtconmatrix)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(gbdtconmatrix)[0] * 100))
print('Gradient Boosting Model Accuracy = %.3f%%' %(gbdtscores * 100))
params_search = {'n_estimators':[10, 60, 100], 
                 'learning_rate': [0.4, 0.6],
                 'max_depth':list(range(2,9)), 
                 'min_samples_split':[100,300,500]}
gbdtmodel_paras = GridSearchCV(gbdtmodel, params_search, n_jobs = -1)
gbdtfitted_gs = gbdtmodel_paras.fit(X_train, y_train)
gbdtscores_gs = gbdtfitted_gs.score(X_test, y_test)
gbdtpredictions_gs = gbdtfitted_gs.predict(X_test)
gbdtconmatrix_gs = metrics.confusion_matrix(y_test, gbdtpredictions_gs)
print('GBDT GridSearchCV Confusion Matrix:', '\n', gbdtconmatrix_gs)
print('best seller to best seller predicted = %.3f%%' %(get_accuracy(gbdtconmatrix_gs)[1] * 100))
print('not best seller to not best seller predicted = %.3f%%' %(get_accuracy(gbdtconmatrix_gs)[0] * 100))
print('Gradient Boosting Model GridSearchCV Accuracy = %.3f%%' %(gbdtscores_gs * 100))
q_GBDT = PrettyTable()
q_GBDT.field_names = ['Comparision', 'True positive rate', 'True negative rate', 'Model Accuracy']
q_GBDT.add_row(['GBDT', 
           '%0.3f%%'%(get_accuracy(gbdtconmatrix)[1] * 100), 
           '%0.3f%%'%(get_accuracy(gbdtconmatrix)[0] * 100), 
           '%0.3f%%'%(gbdtscores * 100)])
q_GBDT.add_row(['GBDT + GridSearch', 
           '%0.3f%%'%(get_accuracy(gbdtconmatrix_gs)[1] * 100), 
           '%0.3f%%'%(get_accuracy(gbdtconmatrix_gs)[0] * 100), 
           '%0.3f%%'%(gbdtscores_gs * 100)])
print(q_GBDT, '\n')
