# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
~~~
## Name: Ritesh DP
## Reg No:212225040339
~~~
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start by importing the required libraries and loading the employee dataset.

2.Preprocess the dataset by converting categorical data into numerical values and splitting it into training and testing data.

3.Create and train the Decision Tree Classifier model using the training dataset.

4.Predict whether employees will stay or leave and evaluate the model using accuracy and confusion matrix.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S Madhumitha
RegisterNumber: 212225040217 
*/
import pandas as pd
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

uploaded = files.upload()

data = pd.read_csv('/content/Employee (1).csv')

print(data.head())

data = pd.get_dummies(data, drop_first=True)

X = data.drop('left', axis=1)
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion='entropy', random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(15, 8))
plot_tree(model, feature_names=X.columns,
          class_names=['Stay', 'Leave'],
          filled=True, max_depth=3)
plt.show()
```

## Output:
<img width="1920" height="1020" alt="Screenshot 2026-11-17 085753" src="https://github.com/user-attachments/assets/6cc48358-2f85-4c63-96f4-8be39dee5770" />
<img width="1920" height="1020" alt="Screenshot 2026-11-17 085804" src="https://github.com/user-attachments/assets/80d8c61a-295b-4952-ae6e-2e76a4f36106" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
