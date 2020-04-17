# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:14:03 2020

@author: login
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sb

path="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv"
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)

#Linear Regression (Yhat=a+bX)
#Create Linear Regression Object
lm = LinearRegression()
print(lm)

"""
# Making Independent and Dependent Variables
X = df[["highway-mpg"]]
Y=df["price"]
#Fit the Linear Model
lm.fit(X, Y)
#Predicting Output
Yhat = lm.predict(X)
print(Yhat[:5])
#Value of Intercept (a)
print("Intercept is: ",lm.intercept_)
#Value of Coefficient (b)
print("Coefficient is: ",lm.coef_)
#Final Linear Model with Structure (Yhat = a+bX)
#price= lm.intercept_-lm.coef_(df["highway-mpg"])
"""
"""
#Train the model using Enigne size as Independent and Price as a Dependent Varibale
x=df[["engine-size"]]
y=df["price"]
#Or lm.fit(df[["engine-size"]],df["price"])
lm.fit(x,y)
yhat=lm.predict(x)
print(yhat[:5])
print("Intercept is: ",lm.intercept_)
print("Slope is: ",lm.coef_)
"""
"""
#Multiple Linear Regression 
#Yhat = a+b1X1+b2X2+b3X3.................
z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
#Fit the Muliple Linear Model using the above variables
lm.fit(z,df["price"])
#Value of Intercept (a)
print("Intercept is: ",lm.intercept_)
#Value of Coefficient (b)
print("Coefficient is: ",lm.coef_)
#Predicting the Output
Yhat=lm.predict(z)
print(Yhat[:5])
"""
"""
lm2 = LinearRegression()
print(lm2)
lm2.fit(df[["normalized-losses","highway-mpg"]], df["price"])
print("Intercept is: ",lm2.intercept_)
print("Coefficent is: ",lm2.coef_)
Yhat=lm2.predict(df[["normalized-losses","highway-mpg"]])
print(Yhat[:5])
"""

#Model Evaluation Using Visualization
#Regression Plot
"""
#Setting Height and Width of Plot
height = 10
width=12
#Setting Plot Width and Height
plt.figure(figsize=(width,height))
#Ploting plot
sb.regplot(x="highway-mpg",y="price", data=df)
#Setting Y axis to 0
plt.ylim(0,)
"""
"""
#Making a Regression Plot of Peak-RPM and Price
plt.figure(figsize=(12,10))
sb.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

#Finding the Correlation of price against highway-mpg and peak-rpm
print(df[["highway-mpg","peak-rpm","price"]].corr())
"""
"""
#Residual Plots
wd=12
ht=10
plt.figure(figsize=(wd,ht))
sb.residplot(x="highway-mpg", y="price", data=df)
plt.show()
"""
"""
#Multiple Linear Regression
z=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(z,df["price"])
Yhat=lm.predict(z)
#Distribution plots
#Setting Plot Size
plt.figure(figsize=(12,10))
#Making Plot of Price (Actual Values)
ax1=sb.distplot(df["price"],hist=False,color='g',label="Actual Value")
#Making Plot of Yhat (Fitted Values)
sb.distplot(Yhat,hist=False,color="r",label="Fitted Values", ax=ax1)
#Setting Title, X and Y Labels
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
#Showing the Plot (Not Necessary in Sypder)
plt.show()
#Closing the Plot
plt.close()
"""
"""
#Ploynomial Regression
#Using a Funtion
def PlotPolly(model, independent_var, dependent_var, name):
    x_new=np.linspace(15,55,100)
    y_new=model(x_new)
    plt.plot(independent_var, dependent_var, ".", x_new, y_new, "-")
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig=plt.gcf()
    plt.xlabel(name)
    plt.ylabel('Price of Cars')
    plt.show()
    plt.close()
    

x = df["highway-mpg"]
y=df["price"]

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x,y,3)
p = np.poly1d(f)
print(p)
#Plot the Above Function
PlotPolly(p,x,y,"highway-mpg")
print(np.polyfit(x,y,3))

#Printing a 11 Order Polynomial Function
f = np.polyfit(x,y,11)
p=np.poly1d(f)
print(p)
PlotPolly(p, x, y, "highway-mpg")

#Ploynomial Features
#Create an object oof polynomial features
pr=PolynomialFeatures()
print(pr)
z=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
z_pr = pr.fit_transform(z)
print(z.shape)
print(z_pr.shape)

#Piplines
#Create PipeLine
input=[("scale", StandardScaler()), 
       ("polynomial", PolynomialFeatures(include_bias=False)),
       ("model", LinearRegression())]
#Input data to Pipeline Constructor
pipe=Pipeline(input)
#Print Pipeline
print(pipe)
#Normalize Data and Fit the Model Simultaneously
pipe.fit(z,y)
#normalize the data, perform a transform and produce a prediction simultaneously
ypipe=pipe.predict(z)
print(ypipe[:5])

#Create a pipeline that Standardizes the data, 
#then perform prediction using a linear regression model using the features Z 
#and targets y
input=[("scale", StandardScaler()), ("model",LinearRegression())]
pipe=Pipeline(input)
pipe.fit(z,y)
ypipe=pipe.predict(z)
print(ypipe[:5])
"""
#Measures for In-Sample Evaluation
"""
#Calculating R^2 for Simple Linear Regression
lm.fit(df[["highway-mpg"]], df["price"])
print("R^2 is: ",lm.score(df[["highway-mpg"]], df["price"]))
#Calculating Mean square Error
Yhat=lm.predict(df[["highway-mpg"]])
print("Yhat is: ",Yhat[:5])
#Calculate MSE
mse = mean_squared_error(df["price"], Yhat)
print("Mean Square error of price and Predicted Value is: ", mse)
"""
"""
#For Multiple Linear Regression
z=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#Fiting the Model
lm.fit(z,df["price"])
print("R^2 is: ",lm.score(z,df["price"]))
#Prediction of Multilinear Model
y_predict_multifit = lm.predict(z)
#Mean Square Error of Multilinear Model
print("Mean Square Error of price and predicted value using Multifit is: ",
      mean_squared_error(df["price"], y_predict_multifit))
"""
"""
#Polynomial Fit
#Calculate R^2 Error
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
#calculate the MSE:
mean_squared_error(df['price'], p(x))
"""
""""
#Prediction and Decision Making
#Fit the model
new_input=np.arange(1,100,1).reshape(-1,1)
lm.fit(df[["highway-mpg"]], df["price"])
print(lm)
#Produce a prediction
yhat=lm.predict(new_input)
print("Prediction is:",yhat[0:5])
#plot the data
plt.plot(new_input, yhat)
plt.show()
"""