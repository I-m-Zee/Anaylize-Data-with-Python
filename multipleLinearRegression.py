# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:13:53 2020

@author: login
"""

#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import pylab as lp
import numpy as np
from sklearn import linear_model


#File Address
file_Address="F:\\KamyabJawan Program\\Machine Learning\\Practice\\Files\\FuelConsumption.csv"
#Reading File
df=pd.read_csv(file_Address)
#Printing Files Rows
print(df.head(10))
#Lets Select Some Features
some_features=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY',
                  'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#Printing Rows of Some Features
print(some_features.head(10))

"""
#Ploting CO2Emission vs Engine Size
plt.scatter(df[["ENGINESIZE"]], df[["CO2EMISSIONS"]], color="blue")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.title("CO2 Emission vs Engine Size")
"""

#Create Train and Test Data sets
msk=np.random.rand(len(df))<0.8
train=some_features[msk]
test=some_features[~msk]


#Ploting Train Data Distribution
plt.scatter(train[["ENGINESIZE"]], train[["CO2EMISSIONS"]], color="blue")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.title("Train Data Distribution")

#Ploting Test Data Distribution
plt.scatter(test[["ENGINESIZE"]], test[["CO2EMISSIONS"]], color="y")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.title("Test Data Distribution")




#Creating Multiple Linear Regression Model
regr=linear_model.LinearRegression()
x=np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
y=np.asanyarray(train.CO2EMISSIONS)
regr.fit(x, y)
#Printing Coefficients
print("Coefficeints: ",regr.coef_)
print("Intercept: ",regr.intercept_)

#Prediction (Testing Data Set to predict Output)
y_hat=regr.predict(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
x=np.asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
y=np.asanyarray(test[["CO2EMISSIONS"]])
#Printing Residual Sum of Square
print("Residual Sum of Square %.2f"% np.mean(y_hat-y)**2)
#Explained Variace Score. 1 is Best Prediction
print("Variance Score: %.2f"% regr.score(x, y))


"""
𝚎𝚡𝚙𝚕𝚊𝚒𝚗𝚎𝚍𝚅𝚊𝚛𝚒𝚊𝚗𝚌𝚎(𝑦,𝑦̂ )=1−(𝑉𝑎𝑟{𝑦−𝑦̂ }/𝑉𝑎𝑟{𝑦})
"""