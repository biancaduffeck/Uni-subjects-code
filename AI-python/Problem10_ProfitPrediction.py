import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

"""0) Read the dataset into pandas dataframe paying attention to file delimeter."""
df = pd.read_csv('50_Startups.csv',delimiter=",")

"""1) Identify the variables inside the dataset"""
#R&D Spend,Administration,Marketing Spend,State,Profit

"""2) Investigate the correlation between the variables"""
sns.heatmap(data=df.corr(numeric_only=True).round(2), annot=True)
plt.show()


"""3) Choose appropriate variables to predict company profit. Justify your choice.
I chose R&D Spend and Marketing Spend because the correlation number in the heat map was .97 (R&D Spend) and .75(Marketing Spend) to the profit,
so I belive they are good variables to predict the profit"""

x = df[['R&D Spend', 'Marketing Spend']]
y = df[['Profit']]

"""4) Plot explanatory variables against profit in order to confirm (close to) linear dependence"""
plt.subplot(2, 2, 1)
plt.xlabel("Profit")
plt.ylabel("R&D Spend")
plt.scatter(df[['Profit']], df[['R&D Spend']], color='blue')
plt.subplot(2, 2, 2)
plt.xlabel("Profit")
plt.ylabel("Marketing Spend")
plt.scatter(df[['Profit']], df[['Marketing Spend']], color='blue')
plt.show()

"""5) Form training and testing data (80/20 split)"""

xpd, xTest, ypd, yTest = train_test_split(x,y,test_size=0.2)

"""6) Train linear regression model with training data"""
regr = linear_model.LinearRegression()
regr.fit(xpd, ypd)

xval = xpd
yval = regr.predict(xval)

"""7) Compute RMSE and R2 values for training and testing data separately"""
print("==================================================================================")
print("R&D Spend, Marketing Spend")
print("==================================================================================")
print("Training Values")
print("RMSE =", np.sqrt(mean_squared_error(yval,ypd)))
print("r2 =",r2_score(yval,ypd))

print("==================================================================================")
xvalTest=xTest
yvalTest=regr.predict(xvalTest)
print("Test Values")
print("RMSE =", np.sqrt(mean_squared_error(yTest,yvalTest)))
print("r2 =",r2_score(yTest,yvalTest))
print("==================================================================================")

"""Since R&D have much more corellation than Marketing Spend, I decide to run a test using only one variable, I noticed that it does not
perform better, but the diference is not that relevant too, for exemple, in one of my compilations, the R2 diference in test values was 0.0288, and in training was even lower(0.003). """

x = df[['R&D Spend', 'Marketing Spend']]
y = df[['Profit']]

xpd, xTest, ypd, yTest = train_test_split(x,y,test_size=0.2)

regr = linear_model.LinearRegression()
regr.fit(xpd, ypd)

xval = xpd
yval = regr.predict(xval)

print("==================================================================================")
print("R&D Spend")
print("==================================================================================")
print("Training Values")
print("RMSE =", np.sqrt(mean_squared_error(yval,ypd)))
print("r2 =",r2_score(yval,ypd))

print("==================================================================================")
xvalTest=xTest
yvalTest=regr.predict(xvalTest)
print("Test Values")
print("RMSE =", np.sqrt(mean_squared_error(yTest,yvalTest)))
print("r2 =",r2_score(yTest,yvalTest))
print("==================================================================================")