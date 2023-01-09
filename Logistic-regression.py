import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

training_dataset = pd.read_csv('training-set-ml.csv')
pd.options.display.max_columns = len(training_dataset.columns)
print(training_dataset)

# training_dataset['Gender'].value_counts().plot.bar(figsize=(20,10), title='Gender')
# plt.show()
# training_dataset['Married'].value_counts().plot.bar(title='Married')
# plt.show()
# training_dataset['Self_Employed'].value_counts().plot.bar(title='Self_Employed')
# plt.show()
# training_dataset['Credit_History'].value_counts().plot.bar(title='Credit_History')
# plt.show()


# training_dataset['Dependents'].value_counts().plot.bar(figsize=(24,6), title='Dependents',color="red")
# plt.show()
# training_dataset['Education'].value_counts().plot.bar(title='Education')
# plt.show()
# training_dataset['Property_Area'].value_counts().plot.bar(title='Property_Area')
# plt.show()


# training_dataset['ApplicantIncome'].plot.box(figsize=(16,5))
# plt.show()
# training_dataset['CoapplicantIncome'].plot.box(figsize=(16,5))
# plt.show()
# training_dataset['LoanAmount'].plot.box(figsize=(16,5))
# plt.show()
# training_dataset['Loan_Amount_Term'].plot.box(figsize=(16,5))
# plt.show()

# categorical features mapped to targeted labels
# Gender=pd.crosstab(training_dataset['Gender'],training_dataset['Loan_Status'])
# Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
# plt.show()
# Married=pd.crosstab(training_dataset['Married'],training_dataset['Loan_Status'])
# Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
# plt.show()
# Self_Employed=pd.crosstab(training_dataset['Self_Employed'],training_dataset['Loan_Status'])
# Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
# plt.show()
# Credit_History=pd.crosstab(training_dataset['Credit_History'],training_dataset['Loan_Status'])
# Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
# plt.show()

#ordinal features mapped to targeted labels
# Dependents=pd.crosstab(training_dataset['Dependents'],training_dataset['Loan_Status'])
# Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
# plt.show()
# Education=pd.crosstab(training_dataset['Education'],training_dataset['Loan_Status'])
# Education.div(Education.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
# plt.show()
# Property_Area=pd.crosstab(training_dataset['Property_Area'],training_dataset['Loan_Status'])
# Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
# plt.show()

#numerical features mapped to targeted labels
# bins=[0,2500,4000,6000,81000]
# group=['Low’','Average’','High','Very high']
# training_dataset['Income_bin']=pd.cut(training_dataset['ApplicantIncome'],bins,labels=group)
# Income_bin=pd.crosstab(training_dataset['Income_bin'],training_dataset['Loan_Status'])
# Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
# plt.xlabel('ApplicantIncome')
# P=plt.ylabel('Percentage')
# plt.show()
#
# bins=[0,100,200,700]
# group=['Low','Average','High']
# training_dataset['LoanAmount_bin']=pd.cut(training_dataset['LoanAmount'],bins,labels=group)
# LoanAmount_bin=pd.crosstab(training_dataset['LoanAmount_bin'],training_dataset['Loan_Status'])
# LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
# plt.xlabel('LoanAmount')
# P=plt.ylabel('Percentage')
# plt.show()
#
# bins=[0,100,200,700]
# training_dataset['Loan_Amount_Term_bin']=pd.cut(training_dataset['Loan_Amount_Term'],bins)
# LoanAmount_bin=pd.crosstab(training_dataset['Loan_Amount_Term_bin'],training_dataset['Loan_Amount_Term'])
# LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
# plt.xlabel('Loan_Amount_Term')
# P=plt.ylabel('Percentage')
# plt.show()

# bins=[0,1000,3000,42000]
# group=['Low','Average','High']
# training_dataset['Coapplicant_Income_bin']=pd.cut(training_dataset['CoapplicantIncome'],bins,labels=group)
# Coapplicant_Income_bin=pd.crosstab(training_dataset['Coapplicant_Income_bin'],training_dataset['Loan_Status'])
# Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
# plt.xlabel('CoapplicantIncome')
# P=plt.ylabel('Percentage')
# plt.show()

# print(training_dataset.isnull().sum())
training_dataset['Gender'].fillna(training_dataset['Gender'].mode()[0], inplace=True)
training_dataset['Married'].fillna(training_dataset['Married'].mode()[0], inplace=True)
training_dataset['Dependents'].fillna(training_dataset['Dependents'].mode()[0], inplace=True)
training_dataset['Self_Employed'].fillna(training_dataset['Self_Employed'].mode()[0], inplace=True)
training_dataset['Credit_History'].fillna(training_dataset['Credit_History'].mode()[0], inplace=True)
training_dataset['Loan_Amount_Term'].fillna(training_dataset['Loan_Amount_Term'].mode()[0], inplace=True)
training_dataset['LoanAmount'].fillna(training_dataset['LoanAmount'].median(), inplace=True)
# print(training_dataset.isnull().sum())

# training_dataset = training_dataset.drop('Loan_ID', axis=1)
# testing_dataset = testing_dataset.drop('Loan_ID', axis=1)




train=training_dataset.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
y = train.Loan_Status
X = pd.get_dummies(X)
train=pd.get_dummies(train)



         
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scal = scaler.fit_transform(X)

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scal,y, test_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.25)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)                                                                                       
print("Model accuracy: ",model.score(x_train,y_train))                                                     
pred_test = model.predict_proba(x_test)                                                                     
print("Discrepancy between prediction and testing actual after training: ",log_loss(y_test,pred_test))
pred_cv = model.predict_proba(x_val)                                                                                
print("Discrepancy between predicted and actual after testing: ", log_loss(y_val,pred_cv))








