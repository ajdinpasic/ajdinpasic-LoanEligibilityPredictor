from json import encoder

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

training_dataset = pd.read_csv("training-set-ml.csv");
pd.options.display.max_columns = len(training_dataset.columns)
# print(training_dataset)
#
# print(training_dataset.isnull().sum())
training_dataset['Gender'].fillna(training_dataset['Gender'].mode()[0], inplace=True)
training_dataset['Married'].fillna(training_dataset['Married'].mode()[0], inplace=True)
training_dataset['Dependents'].fillna(training_dataset['Dependents'].mode()[0], inplace=True)
training_dataset['Self_Employed'].fillna(training_dataset['Self_Employed'].mode()[0], inplace=True)
training_dataset['Credit_History'].fillna(training_dataset['Credit_History'].mode()[0], inplace=True)
training_dataset['Loan_Amount_Term'].fillna(training_dataset['Loan_Amount_Term'].mode()[0], inplace=True)
training_dataset['LoanAmount'].fillna(training_dataset['LoanAmount'].median(), inplace=True)

train=training_dataset.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
y = train.Loan_Status
X = pd.get_dummies(X)
train=pd.get_dummies(train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.metrics import log_loss
x_scal = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(x_scal,y, test_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.25)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

svc = svm.SVC(C=1, kernel='linear')
X_train, X_test, y_train, y_test = train_test_split( x_scal, y, test_size=0.4, random_state=0)
clf = svc.fit(x_scal, y)
predicted = cross_val_predict(clf, x_scal, y, cv=2)

y_pred_num  = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred_num)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()

print("Accuracy",metrics.accuracy_score(y, predicted))
print("Precision",metrics.precision_score(y, predicted, pos_label="Y"))
print("Sensitivity",metrics.recall_score(y, predicted, pos_label="N"))
print("Specificity",metrics.recall_score(y, predicted,  pos_label="Y"))
