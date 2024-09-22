# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

5.Generate Confusion Matrix
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Starbiya S
RegisterNumber:212223040208  
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(cm)
```

## Output:
![Screenshot 2024-09-22 185515](https://github.com/user-attachments/assets/91164eab-d80f-4a69-a6d8-fe729b7a7c5e)


![Screenshot 2024-09-22 185522](https://github.com/user-attachments/assets/f5c510da-008f-4b67-af58-3cecd46b8057)


![Screenshot 2024-09-22 185529](https://github.com/user-attachments/assets/7b909605-dcff-41ec-b025-b6f900127f32)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
