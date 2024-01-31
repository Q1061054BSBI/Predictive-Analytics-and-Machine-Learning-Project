import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

print("Start. Training of the model...")
#import
df=pd.read_csv('Bank_Personal_Loan_Modelling.csv');

#cleaning & preprocessing
if 'Experience' in df :
    df.drop('Experience',axis=1,inplace=True)
if 'ID' in df:
    df.drop('ID',axis=1,inplace=True)
df.isnull().sum()
df.drop_duplicates(inplace=True)

#division into test and training sample
X=df.drop(['Personal Loan'],axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state = 42)

# RF
regrf = RandomForestClassifier(random_state=42)
params = {
    'max_depth': [2,3,5],
    'min_samples_leaf': [5,10,20],
    'n_estimators': [50,250, 300 ]
}

# searching for the best hyperparameters
gridsearch_rf = GridSearchCV(estimator=regrf,  param_grid =params, cv = 4, n_jobs=-1, verbose=1, scoring = 'f1')
gridsearch_rf.fit(X_train, Y_train)

#Data entry
print('Model is ready. Enter your data.')
print()
print("In case of binary values (YES/NO), enter 1 or 0. In case in numeric value, enter a number")

NEW_X = pd.DataFrame()

for keys,col in enumerate(X):
    data = float(input("Enter "+col+": "))
    array = np.array([data])
    NEW_X[col] = pd.Series(array)
    print(type(NEW_X[col]))    


#Result calculation
result = gridsearch_rf.predict(NEW_X);
print("Predicting the result (0 - passive, 1 - active):")
print(result[0])




