# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
step-1: start.
step-2: Import chardet.
step-3: Read the dataset.
step-4: Import SVC from sklearn.
step-5: Fit the data in the model and run the algorithm.
step-6: stop.
```
## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Krithiga U
RegisterNumber:  212223240076

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
data.head()

![image](https://github.com/user-attachments/assets/d76db296-d309-492d-b794-bc69a71fb5f9)

data.tail()

![image](https://github.com/user-attachments/assets/55bca5b1-94dc-4686-ad4b-e4dc452ead22)

data.info()

![image](https://github.com/user-attachments/assets/3a481861-2f2c-4ee4-b98f-8f926f5f3a47)

y_pred value

![image](https://github.com/user-attachments/assets/2001f8b6-17e4-42a3-b058-99ea863d3bdd)

Accuracy value
![image](https://github.com/user-attachments/assets/e95c6f76-60f6-4af3-bc7b-9019c37d6b02)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
