# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:26:40 2020

@author: I'm Zee
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats
#Importing File
file_Address="F:\\KamyabJawan Program\\Machine Learning\\Practice\\Files\\automobileEDA.csv"
df=pd.read_csv(file_Address)
#Printing File 5 rows
print(df.head())

#Data type of each columns
print(df.dtypes)

#Calculation of Correlation
print(df.corr())

#Find Correlation of Specified COlumns
print(df[["bore","stroke","compression-ratio","horsepower"]].corr())
"""
#Scatter Plot of Engine Size and Price
sbn.regplot(x="engine-size",y="price",data=df)
plt.ylim(0,)
plt.xlim(0,)
"""
#Correletion of Engine Size and Price
print(df[["engine-size","price"]].corr())

#Correletion oof Highway Miles and price
print(df[["highway-mpg","price"]].corr())
"""
#Scatter Plot of Highway-MPG and Price
sbn.regplot(x="highway-mpg",y="price",data=df)
"""

#Correletion of Peak-RPM and Price
print(df[["peak-rpm","price"]].corr())
"""
#Scatter Plot of Peak-RPM and Price
sbn.regplot(x="peak-rpm",y="price",data=df)
"""
"""
#Correletion and Scatter Plot of Stroke and Price
print(df[["stroke","price"]].corr())
sbn.regplot(x="stroke",y="price",data=df)
"""
"""
#Categorical Visulazetion of Body Style and Price
sbn.boxplot(x="body-style",y="price",data=df)
"""
"""
#Categorical Visulazetion of Engine Location and Price
sbn.boxplot(x="engine-location",y="price",data=df)
"""
"""
#Categorical Visulazetion of Engine Size and Price
sbn.boxplot(x="engine-size",y="price",data=df)
"""
"""
#Categorical Visulazition of Drive WHeels and Price
sbn.boxplot(x="drive-wheels",y="price",data=df)
"""
#Descriptive Statistical Analysis
print(df.describe())

#To Include the Object type variables
print(df.describe(include=["object"]))
print(df.describe(exclude=["int64"]))
"""
#Count valued in specific Columns
print(df["drive-wheels"].value_counts())

#Convert a Drive Wheel Series to Data Frame as fellows
print(df["drive-wheels"].value_counts().to_frame())
"""
drive_wheels_count=df["drive-wheels"].value_counts().to_frame()
#Rename the Column
drive_wheels_count.rename(columns={"drive-wheels":"value-count"}, inplace=True)
#Rename Index
drive_wheels_count.index.name="drive-wheels"
print(drive_wheels_count)

#Covnert a Engine Location Series to Data Frame
engine_loc_count=df["engine-location"].value_counts().to_frame()
engine_loc_count.rename(columns={"engine-location":"value-count"},inplace=True)
engine_loc_count.index.name="engine-location"
print(engine_loc_count)

#Grouping
print(df["drive-wheels"].unique())
#Group by Derive Wheels Type
df_group_one=df[["drive-wheels","body-style","price"]]
df_group_one=df_group_one.groupby(["drive-wheels"], as_index=False).mean()
print(df_group_one)

#Group by Drive WHeel and Body Style
df_group_two=df[["drive-wheels","body-style","price"]]
group_test=df_group_two.groupby(["drive-wheels","body-style"], as_index=False).mean()
print(group_test)

#Making a Pivot Table
group1_pivot=group_test.pivot(index="drive-wheels", columns="body-style")
#Filling NAN with 0
group1_pivot=group1_pivot.fillna(0)
print(group1_pivot)
"""
#Make Heatmap to visulize the Data
plt.pcolor(group1_pivot,cmap="twilight")
plt.colorbar()
plt.show()
"""
"""
#Making Labels for Better Understanding
fig, ax = plt.subplots()
im = ax.pcolor(group1_pivot, cmap='RdBu')

#label names
row_labels = group1_pivot.columns.levels[1]
col_labels = group1_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group1_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group1_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
"""
#Calculating Pearson Correlation
print(df.corr())

#Calculating Pearson COefficient and P Value Horse Power and Price
pearson_coeff,p_value=stats.pearsonr(df["horsepower"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)
#Calculating Pearson COefficient and P Value Length and Price
pearson_coeff,p_value=stats.pearsonr(df["length"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)
#Calculating Pearson COefficient and P Value Length and Price
pearson_coeff,p_value=stats.pearsonr(df["width"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)
#Calculating Pearson COefficient and P Value Length and Price
pearson_coeff,p_value=stats.pearsonr(df["curb-weight"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)
#Calculating Pearson COefficient and P Value Length and Price
pearson_coeff,p_value=stats.pearsonr(df["engine-size"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)
#Calculating Pearson COefficient and P Value Length and Price
pearson_coeff,p_value=stats.pearsonr(df["bore"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)
#Calculating Pearson COefficient and P Value Length and Price
pearson_coeff,p_value=stats.pearsonr(df["city-mpg"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)
#Calculating Pearson COefficient and P Value Length and Price
pearson_coeff,p_value=stats.pearsonr(df["highway-mpg"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coeff, 
      " with a P-value of P = ", p_value)

#Analysis of Variance
group_test2=df_group_two[["drive-wheels","price"]].groupby(["drive-wheels"])
print(group_test2.head(2))
print(df_group_two)
#Obtian Group wise Value
print(group_test2.get_group('4wd')['price'])
# ANOVA
f_val, p_val = stats.f_oneway(group_test2.get_group('fwd')['price'], 
                              group_test2.get_group('rwd')['price'], 
                              group_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)

#Fwd and Rwd only
f_val, p_val = stats.f_oneway(group_test2.get_group('fwd')['price'],
                              group_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )
#4wd and Rwd
f_val, p_val = stats.f_oneway(group_test2.get_group('4wd')['price'],
                              group_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)
#4wd and Fwd
f_val, p_val = stats.f_oneway(group_test2.get_group('4wd')['price'],
                              group_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)   
