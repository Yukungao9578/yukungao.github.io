#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:26:31 2019

@author: xinyancai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 

df = pd.read_csv('~/Desktop/data/heloc_dataset_v1.csv')
df.RiskPerformance[df.RiskPerformance == 'Good'] = -1
df.RiskPerformance[df.RiskPerformance == 'Bad'] = -2
da = df.loc[~(df < 0).all(axis=1)]
da.RiskPerformance[da.RiskPerformance == -1] = 1
da.RiskPerformance[da.RiskPerformance == -2] = 0
da[da < 0] = np.nan
data = da.drop(['MSinceMostRecentDelq','MSinceMostRecentInqexcl7days','NetFractionInstallBurden'],axis=1)
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
risk = train_set.copy().drop("RiskPerformance", axis=1)
risk_labels = train_set["RiskPerformance"].copy()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'mean')
risk_prepared = imputer.fit_transform(risk)
X = data.drop("RiskPerformance", axis=1)
Y = data["RiskPerformance"].copy()
X_prepared = imputer.transform(X)
risk_prepared = risk_prepared.astype('int')
risk_labels = risk_labels.astype('int')

import pickle
import warnings
pickle.dump(risk_prepared, open('risk_prepared.sav','wb'))
pickle.dump(X_prepared, open('X_prepared.sav', 'wb'))
pickle.dump(Y, open('Y.sav', 'wb'))
import streamlit as st
from sklearn import metrics
pickle.load(open('risk_prepared.sav','rb'))
pickle.load(open('X_prepared.sav', 'rb'))
pickle.load(open('Y.sav', 'rb'))

dic = {0: 'Bad', 1: 'Good'}

def test_demo(index):
    values = X_prepared[index]  # Input the value from dataset

    # Create four sliders in the sidebar
    a1 = st.sidebar.slider('ExternalRiskEstimate', 0.0, 110.0, values[0], 0.1)
    a2 = st.sidebar.slider('MSinceOldestTradeOpen', 0.0, 810.0, values[1], 0.1)
    a3 = st.sidebar.slider('MSinceMostRecentTradeOpen', 0.0, 400.0, values[2], 0.1)
    a4 = st.sidebar.slider('AverageMInFile', 0.0, 400.0, values[3], 0.1)
    a5 = st.sidebar.slider('NumSatisfactoryTrades', 0.0, 110.0, values[4], 0.1)
    a6 = st.sidebar.slider('NumTrades60Ever2DerogPubRech', 0.0, 20.0, values[5], 0.1)
    a7 = st.sidebar.slider('NumTrades90Ever2DerogPubRec', 0.0, 20.0, values[6], 0.1)
    a8 = st.sidebar.slider('PercentTradesNeverDelq', 0.0, 110.0, values[7], 0.1)
    a9 = st.sidebar.slider('MaxDelq2PublicRecLast12Mh', 0.0, 10.0, values[8], 0.1)
    a10 = st.sidebar.slider('MaxDelqEver', 0.0, 10.0, values[9], 0.1)
    a11 = st.sidebar.slider('NumTotalTrades', 0.0, 110.0, values[10], 0.1)
    a12 = st.sidebar.slider('NumTradesOpeninLast12M', 0.0, 20.0, values[11], 0.1)
    a13 = st.sidebar.slider('PercentInstallTrades', 0.0, 110.0, values[12], 0.1)
    a14 = st.sidebar.slider('NumInqLast6M', 0.0, 70.0, values[13], 0.1)
    a15 = st.sidebar.slider('NumInqLast6Mexcl7days', 0.0, 70.0, values[14], 0.1)
    a16 = st.sidebar.slider('NetFractionRevolvingBurden', 0.0, 240.0, values[15], 0.1)
    a17 = st.sidebar.slider('NumRevolvingTradesWBalance', 0.0, 40.0, values[16], 0.1)
    a18 = st.sidebar.slider('NumInstallTradesWBalance', 0.0, 25.0, values[17], 0.1)
    a19 = st.sidebar.slider('NumBank2NatlTradesWHighUtilization', 0.0, 20.0, values[18], 0.1)
    a20 = st.sidebar.slider('PercentTradesWBalance', 0.0, 110.0, values[19], 0.1)

    #Print the prediction result
    alg = ['Random Forest', 'Linear Model', 'Support Vector Machine', 'Bagging','Boosting','Neural Network' ]
    classifier = st.selectbox('Which algorithm?', alg)
    if classifier == 'Random Forest':
        # different trained models should be saved in pipe with the help pickle
        
        from sklearn.ensemble import RandomForestRegressor
        param_grid = [{'n_estimators':[3,10,20,30],'max_features':[2,4,6,8]},
              {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}]
        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(risk_prepared,risk_labels)
        cvres = grid_search.cv_results_ # the variable that stores the grid search results
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
              print(np.sqrt(-mean_score), params)
        from sklearn.metrics import mean_squared_error
        final_model = grid_search.best_estimator_
        final_predictions = final_model.predict(np.array([a1, a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20]).reshape(1, -1))[0]
        acc = 1 - final_predictions
        st.write('Risk: ', acc)
        
        if final_predictions < 0.5:
            final_predictions = 0
        else: 
            final_predictions = 1
        st.write('Final Predictions: ', dic[final_predictions])
        
        
        st.text('Random Forest Chosen')
        
    elif classifier == 'Linear Model':
         from sklearn import linear_model
         param_grid_linear = [{'C':[1,10,100,10**10]}]
         clf_linear = GridSearchCV(linear_model.LogisticRegression(), param_grid_linear, cv = 5)
         clf_linear.fit(risk_prepared,risk_labels)
         cvres = clf_linear.cv_results_ # the variable that stores the grid search results
         for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
              print(np.sqrt(mean_score), params)
         final_model = clf_linear.best_estimator_
         final_predictions = final_model.predict(np.array([a1, a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20]).reshape(1, -1))[0]
         acc = 1 - final_predictions
         st.write('Risk: ', acc)
        
         if final_predictions < 0.5:
            final_predictions = 0
         else: 
            final_predictions = 1
         st.write('Final Predictions: ', dic[final_predictions])
         
         st.text('Linear Model Chosen')
         
    elif classifier == 'Support Vector Machine':
         from sklearn.svm import SVC
         param_grid_svc = [{'C':[0.01,0.1,1,10],'kernel':['rbf','linear','poly'], 'max_iter':[-1],'random_state':[1]}]
         clf_svm = GridSearchCV(SVC(gamma = 'scale'), param_grid_svc, cv=5)
         clf_svm.fit(risk_prepared,risk_labels)
         cvres = clf_svm.cv_results_ # the variable that stores the grid search results
         for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
              print(np.sqrt(mean_score), params)
         final_model = clf_svm.best_estimator_
         final_predictions = final_model.predict(np.array([a1, a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20]).reshape(1, -1))[0]
         acc = 1 - final_predictions
         st.write('Risk: ', acc)
        
         if final_predictions < 0.5:
            final_predictions = 0
         else: 
            final_predictions = 1
         st.write('Final Predictions: ', dic[final_predictions])
         
         st.text('Support Vector Machine')
         
    elif classifier == 'Bagging':
         from sklearn import tree
         from sklearn.ensemble import BaggingClassifier
         base1 = tree.DecisionTreeClassifier(max_depth = 1)
         base2 = tree.DecisionTreeClassifier(max_depth = 10)
         base3 = tree.DecisionTreeClassifier(max_depth = 20)
         base4 = tree.DecisionTreeClassifier(max_depth = 50)
         param_grid_bagging = [{'n_estimators':[5,10,20,30,50],'base_estimator':[base1,base2,base3,base4]}]
         clf_bagging = GridSearchCV(BaggingClassifier(), param_grid_bagging, cv=5)
         clf_bagging.fit(risk_prepared,risk_labels)
         cvres = clf_bagging.cv_results_ # the variable that stores the grid search results
         for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
              print(np.sqrt(mean_score), params)
         final_model = clf_bagging.best_estimator_
         final_predictions = final_model.predict(np.array([a1, a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20]).reshape(1, -1))[0]
         acc = 1 - final_predictions
         st.write('Risk: ', acc)
        
         if final_predictions < 0.5:
            final_predictions = 0
         else: 
            final_predictions = 1
         st.write('Final Predictions: ', dic[final_predictions])
         
         st.text('Bagging')
    
    elif classifier == 'Boosting':
         from sklearn.ensemble import AdaBoostClassifier
         param_grid_boosting = [{'n_estimators':[5,10,20,30,50],'learning_rate':[0.1,0.5,1,10],'random_state':[1]}]
         clf_boost = GridSearchCV(AdaBoostClassifier(), param_grid_boosting, cv=5)
         clf_boost.fit(risk_prepared,risk_labels)
         cvres = clf_boost.cv_results_ # the variable that stores the grid search results
         for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
              print(np.sqrt(mean_score), params)
         final_model = clf_boost.best_estimator_
         final_predictions = final_model.predict(np.array([a1, a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20]).reshape(1, -1))[0]
         acc = 1 - final_predictions
         st.write('Risk: ', acc)
        
         if final_predictions < 0.5:
            final_predictions = 0
         else: 
            final_predictions = 1
         st.write('Final Predictions: ', dic[final_predictions])
         
         st.text('Boosting')
         
    elif classifier == 'Neural Network': 
         from sklearn.neural_network import MLPClassifier
         param_grid_MLP = [{'hidden_layer_sizes':[(100,)],'activation':['identity','logistic','tanh', 'relu'],
                   'solver': ['lbfgs','sgd','adam'],'alpha':[0.0001,0.001,0.01],'random_state':[1]}]
         clf_MLP = GridSearchCV(MLPClassifier(), param_grid_MLP, cv=5)
         clf_MLP.fit(risk_prepared,risk_labels)
         cvres = clf_MLP.cv_results_ # the variable that stores the grid search results
         for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
              print(np.sqrt(mean_score), params)
         final_model = clf_MLP.best_estimator_
         final_predictions = final_model.predict(np.array([a1, a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20]).reshape(1, -1))[0]
         acc = 1 - final_predictions
         st.write('Risk: ', acc)
        
         if final_predictions < 0.5:
            final_predictions = 0
         else: 
            final_predictions = 1
         st.write('Final Predictions: ', dic[final_predictions])
         
         st.text('Neural Network')
         
         
         
    
         
         

st.title('Credit Risk')
if st.checkbox('show dataframe'):
    st.write(X_prepared)
number = st.text_input('Choose a row of information in the dataset(0~9870)',5)
test_demo(int(number)) 

