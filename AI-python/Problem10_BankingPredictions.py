import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
"""
1) Read in the CSV file using pandas. Pay attention to the file delimeter.
"""
df = pd.read_csv("bank.csv", delimiter=";")

"""
2) Pick data from the following columns to a second dataframe 'df2': y, job, marital, default, housing, poutcome.
"""
df2=df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

"""
3) Convert categorical variables to dummy numerical values using the command
"""
df3 = pd.get_dummies(df2,columns=['y', 'job','marital','default','housing','poutcome'])

"""
4) Produce a heat map of correlation coefficients for all variables in df3. Describe the amount of correlation between the variables in your own words.

get_dummies function separeted the variable that are strings in 1 and 0 multiple variables, for exemple :
marital string that could be 'divorced', 'married' or 'single' became 3 boolean variables: marital_divorced, marital_married and marital_single
so, a lot of variables are related between them, for exemple: if you are single you are divorced or married, so it's a strong negative corellation betweem them

it was not requested at this part of the exercise to include the 'y', but I did because it thought it would make more sense to see this numbers together

besides the effect of the get_dummies, some correlations that I noted:
- poutcome 'success' has .28 of correlation with y 'yes'
- job 'blue collar' has .18 correlation with housing 'yes'
- jog 'student' has .19 correlation with marital 'single'
they are all very week, but those are the strongers of the table

"""
sns.heatmap(data=df3.corr().round(2), annot=True)
plt.show()

"""
5) Select the column called 'y' of df3 as the target variable y, and all the remaining columns for the explanatory variables X.
"""
temp=df3.drop('y_yes', axis=1)
temp=temp.drop('y_no', axis=1)
x = temp
y = df3['y_yes']

"""
6) Split the dataset into training and testing sets with 75/25 ratio.
"""
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.25)

"""
7) Setup a logistic regression model, train it with training data and predict on testing data.
"""
modelLR = LogisticRegression()
modelLR.fit(xTrain, yTrain)
y_pred = modelLR.predict(xTest)

"""
8) Print the confusion matrix (or use heat map if you want) and accuracy score for the logistic regression model.
"""
cnf_matrix = metrics.confusion_matrix(yTest, y_pred)

print("Logistic Regression ----------------------------------------------------------------")
print("Confusion Matrix")
print(cnf_matrix)
accuracy_scoreLR=metrics.accuracy_score(yTest, y_pred)
precision_scoreLR=metrics.precision_score(yTest, y_pred)
recall_scoreLR=metrics.recall_score(yTest, y_pred)
print("Accuracy:", accuracy_scoreLR)
print("Precision:", precision_scoreLR)
print("Recall:", recall_scoreLR)

"""
9) Repeat steps 7 and 8 for k-nearest neighbors model. Use k=3, for example, or experiment with different values.
after trying with k=3, k=5 and k=7, I found that for this model k=7 usually gives better results
"""
"""
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(xTrain, yTrain)
y_pred = classifier.predict(xTest)
print("KNeighbors Classifier 3 ------------------------------------------------------------")
print("Accuracy:", metrics.accuracy_score(yTest, y_pred))
print("Precision:", metrics.precision_score(yTest, y_pred))
print("Recall:", metrics.recall_score(yTest, y_pred))

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(xTrain, yTrain)
y_pred = classifier.predict(xTest)
print("KNeighbors Classifier 5 ------------------------------------------------------------")
print("Accuracy:", metrics.accuracy_score(yTest, y_pred))
print("Precision:", metrics.precision_score(yTest, y_pred))
print("Recall:", metrics.recall_score(yTest, y_pred))
"""

classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(xTrain, yTrain)
y_pred = classifier.predict(xTest)
cnf_matrix = metrics.confusion_matrix(yTest, y_pred)
print("")
print("KNeighbors Classifier 7 ------------------------------------------------------------")
print("Confusion Matrix")
print(cnf_matrix)
accuracy_scoreKN=metrics.accuracy_score(yTest, y_pred)
precision_scoreKN=metrics.precision_score(yTest, y_pred)
recall_scoreKN=metrics.recall_score(yTest, y_pred)
print("Accuracy:", accuracy_scoreKN)
print("Precision:", precision_scoreKN)
print("Recall:", recall_scoreKN)


"""
10) Compare the results between the two models.
usually the Logistic regression is giving better and more consistent results, but below I print the winners of each run.
I have to point that this exercises is having very low recall, which seems to be result of a high number of fake negatives a low number of true positives (comparing one o each other).
But this seems to be expected from the data, because there is way more 'No' than 'Yes' in the product that the bank was trying to sell.
"""
print("")
print("Better Accuracy: ", "Logistic Regression" if accuracy_scoreLR>accuracy_scoreKN else "KNeighbors Classifier")
print("Better Precision: ", "Logistic Regression" if precision_scoreLR>precision_scoreKN else "KNeighbors Classifier")
print("Better Recall: ", "Logistic Regression" if recall_scoreLR>recall_scoreKN else "KNeighbors Classifier")