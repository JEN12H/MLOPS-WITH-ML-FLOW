import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()
X = wine.data
y = wine.target   

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# DEFINING THE MODEL PARAMETERS 
max_depth = 5
n_estimators = 10

# MENTION YOUR EXPERIMENT NAME BELOW 

mlflow.set_experiment("MLOPS-EXP-1")

with mlflow.start_run():
    # CREATING THE MODEL
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)
    
    # MAKING PREDICTIONS
    y_pred = rf.predict(X_test)
    
    # EVALUATING THE MODEL
    accuracy = accuracy_score(y_test,y_pred)
    
    # LOGGING PARAMETERS, METRICS AND MODEL TO MLFLOW
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    # PRINTING THE RESULTS
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    
    