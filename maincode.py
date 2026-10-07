#step1: data collection
import pandas as pd
#reading data
data=pd.read_csv("loan_data_1.csv")
print(data)

#step2: data analysis
print(data.head())#print first five rows or we pass as  argument in()to get desired o/p

print(data.tail())#print last five rows or we pass as argument in()to get desired o/p

print(data.info()) #information about data

print(data.isnull().sum()) #check missing values

print(data.duplicated().sum())#check duplicate vaules

#data preprocessing
#remove irrelevent data
data=data.drop(["Unnamed: 0","Loan_ID"],axis=1) #to delete column axis (column=1 or row=0)
print(data)

# handle missing values mean,median,mode
print(data.isnull().sum())
data["Gender"]=data["Gender"].fillna(data["Gender"].mode()[0])#using mode

data["Dependents"]=data["Dependents"].fillna(data["Dependents"].mode()[0])#using mode
data["Education"]=data["Education"].fillna(data["Education"].mode()[0])#using mode
data["Self_Employed"]=data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])#using mode
data["ApplicantIncome"]=data["ApplicantIncome"].fillna(data["ApplicantIncome"].mean())#using mean
data["CoapplicantIncome"]=data["CoapplicantIncome"].fillna(data["CoapplicantIncome"].mean())#using mean
data["LoanAmount"]=data["LoanAmount"].fillna(data["LoanAmount"].mean())#using mean
data["Loan_Amount_Term"]=data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])#using mode
data["Credit_History"]=data["Credit_History"].fillna(data["Credit_History"].mode()[0])

print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
columns=["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status","Dependents"]
for i in columns:
    data[i]=le.fit_transform(data[i])
print(data)
print(data.info())

#split data into features and target variables
x=data.drop("Loan_Status",axis=1)
y=data["Loan_Status"]

#split data into training and testing datatest
from sklearn.model_selection import train_test_split #split data in 2 parts
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,shuffle=True,random_state=42)# about random data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)

#model train
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#prediction model
y_predict=model.predict(x_test)
print(y_predict)

#model evaluation
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)







