import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

"""1) Read the data into pandas dataframe"""
df = pd.read_csv('Auto.csv',delimiter=",")

"""1) Identify the variables inside the dataset"""
#mpg,cylinders,displacement,horsepower,weight,acceleration,year,origin,name

"""2) Setup multiple regression X and y to predict 'mpg' of cars using all the variables except 'mpg', 'name' and 'origin'"""

x = df[['cylinders', 'displacement','horsepower','weight','acceleration','year']]
y = df[['mpg']]

"""3) Split data into training and testing sets (80/20 split)"""

xpd, xTest, ypd, yTest = train_test_split(x,y,test_size=0.2)

"""
4) Implement both ridge regression and LASSO regression using several values for alpha
I want to use the same alpha for both regressions but as they have diferent magnitudes, I made a list with small and big numbers to test.

the lasso regression is having better results with numbers with order of magnitude .1, and the rigde regression is really dificult to understand the pattern since I
have seing numbers from .1 to 500, so even the limit of my alphas is 1000, it's possible that the number is out of range
"""

alphas=np.append(np.linspace(0.01,10,600),np.linspace(10,1000,600))
"""5) Search optimal value for alpha (in terms of R2 score) by fitting the models with training data and computing the score using testing data"""

scoresLasso = []
for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(xpd, ypd)
    sc = lasso.score(xTest, yTest)
    scoresLasso.append(sc)

scoresRidge = []
for alp in alphas:
    ridge = Ridge(alpha=alp)
    ridge.fit(xpd, ypd)
    sc = ridge.score(xTest, yTest)
    scoresRidge.append(sc)

"""6) Plot the R2 scores for both regressors as functions of alpha"""

plt.subplot(2, 2, 1)
plt.xlabel("alpha")
plt.ylabel("r2 - LASSO")
plt.scatter(alphas,scoresLasso, color='blue')
plt.subplot(2, 2, 2)
plt.xlabel("alpha")
plt.ylabel("r2 - rigde")
plt.scatter(alphas, scoresRidge, color='blue')
plt.show()

"""
7) Identify, as accurately as you can, the value for alpha which gives the best score
I tried to identify it here, but as commented before, it might be that the Rigde number is out of the tested range
"""

best_r2 = max(scoresLasso)
idx = scoresLasso.index(best_r2)
best_alp = alphas[idx]

print("LASSO best r2",best_r2,"best apha",best_alp)

best_r2 = max(scoresRidge)
idx = scoresRidge.index(best_r2)
best_alp = alphas[idx]

print("Ridge best r2",best_r2,"best apha",best_alp)
