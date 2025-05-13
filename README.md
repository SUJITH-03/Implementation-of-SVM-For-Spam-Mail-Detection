# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sujith A
RegisterNumber:  212224230278
*/

import chardet
file=(r'C:\Users\admin\Downloads\spam.csv')
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv(r'C:\Users\admin\Downloads\spam.csv',encoding='Windows-1252')
print(data.head())
print(data.info())
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
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
```

## Output:
## ENCODING DETECTED
![Screenshot 2025-04-21 232423](https://github.com/user-attachments/assets/102b5f40-76b4-4389-84f6-8b081d83f402)
## FIRST FEW ROWS, DATA INFO, MISSING VALUES
![Screenshot 2025-04-21 232432](https://github.com/user-attachments/assets/eb57e1bb-d8b2-499d-a0d1-28193c783037)
## PREDICTED LABELS
![Screenshot 2025-04-21 232438](https://github.com/user-attachments/assets/496d3781-2560-47ed-bc38-a7b50c42435a)
## MODEL ACCURACY
![Screenshot 2025-04-21 232441](https://github.com/user-attachments/assets/cae6295e-3af8-4c82-89d4-ef45896c2122)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
