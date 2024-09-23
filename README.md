# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
<li>STEP 1:Import the standard Libraries.</li>

<li>STEP 2:Set variables for assigning dataset values.</li>

<li>STEP 3:Import linear regression from sklearn.</li>

<li>STEP 4:Assign the points for representing in the graph</li>

<li>STEP 5:Predict the regression for marks by using the representation of the graph.

STEP 6:Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PREM R
RegisterNumber:  212223240124

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```

## Output:

## DATASET:
![image](https://github.com/user-attachments/assets/8d4108dd-134e-40c1-8e07-774b623ea581)
## Y PREDICTED:
![image](https://github.com/user-attachments/assets/bac93b99-9699-41c0-90bd-bc171e2800b2)
## TRAINING SET:
![image](https://github.com/user-attachments/assets/185635db-64c6-4ba9-8bd3-17bc66be668c)
## VALUES OF MSE,MAE,RMSE:
![image](https://github.com/user-attachments/assets/a936a578-a97c-48c2-852a-79a0d714fe68)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
