# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:01:20 2020

@author: login
"""


import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import sklearn.linear_model as sklm
from sklearn.metrics import r2_score as skmr2

#File Path
file_Address="F:\\KamyabJawan Program\\Machine Learning\\Practice\\Files\\FuelConsumption.csv"
#Reading File
df=pd.read_csv(file_Address)
#Printing Data
print(df.head())

#Describing Data
print(df.describe())

#Selecting Some Feature to Explore Data of that Features
some_features=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
#printing
print(some_features.head())

"""
#ploting all feature separately
#Assigning a variable
visualization=some_features[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
#showing through variable
visualization.hist()
"""
#plot each of these features vs the Emission, to see how linear is their relation
"""
#Relation of Fuel Consumption and CO2 Emission
plt.scatter(some_features.FUELCONSUMPTION_COMB, some_features.CO2EMISSIONS, color='blue')
plt.xlabel("Fuel Consumption")
plt.ylabel("CO2 Emission")
plt.title("Relation of CO2 and Fuel Consumption")
"""
"""
#Relation of Engine Size and CO2 Emission
plt.scatter(some_features.ENGINESIZE,some_features.CO2EMISSIONS,color='red')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.title("Relation of CO2 and Engine Size")
"""
"""
#Relation of Clynders and CO2 Emission
plt.scatter(some_features.CYLINDERS, some_features.CO2EMISSIONS, color="green")
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emission")
plt.title("Relation of CO2 and Engine Size")
"""
#Creating Training and Test Data Sets
testData=np.random.rand(len(df))<0.8
train=some_features[testData]
test=some_features[~testData]
print("Training Data Set: ",train)
print("Testing Data Set: ",test)

#Making Simple Regression Model
#Modeling our Data
regr=sklm.LinearRegression()
train_x=np.asanyarray(train[["ENGINESIZE"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x,train_y)
#Printing Coefficiets and Intercepts
print("Coefficient: ",regr.coef_)
print("Intercept: ",regr.intercept_)

#Ploting our Model on Graph
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.plot(train_x,regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.title("SRM of Engine vs Emission")

#Evaluation of Our Model
test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])
test_y_hat=regr.predict(test_x)
plt.plot(test_x,regr.coef_[0][0]*test_x + regr.intercept_[0], 'y')
#printing Values
print("Mean Absolute Error: %.2f"% np.mean(np.absolute(test_y_hat-test_y)))
print("Residual Sum of Squares (MSE): %.2f"% np.mean(test_y_hat-test_y)**2)
print("R2-Score: %.2f"% skmr2(test_y_hat,test_y))
