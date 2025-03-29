from sklearn.tree import DecisionTreeClassifier
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from utility import pipeline

# TODO: Set tht MLFlow tracking server uri
uri = "http://127.0.0.1:6001"
# Use mlflow.set_tracking_uri to set the uri
mlflow.set_tracking_uri(uri)

# Set experiment nam
email = "your_email@andrew.cmu.edu"   # TODO: Use a customized experiment name
experiment_name = f"{email}-lab7"
mlflow.set_experiment(experiment_name)

# TODO: Generates train and test dataset using `pipeline.data_preprocessing` function
X_train, X_test, y_train, y_test = pipeline.data_preprocessing()

params = {
    "criterion": "gini",
    "max_depth": 5,
    "random_state": 42
}

'''
I updated the registered model by using a Decision Tree instead of Logistic Regression. 
I chose a Decision Tree because it can capture more complex patterns in the data, unlike Logistic 
Regression which assumes everything is linear. It’s also easier to understand and visualize, which
helps when explaining how the model makes decisions.
'''
# Train decisition tree
model = DecisionTreeClassifier(**params)
model.fit(X_train, y_train)

accuracy = pipeline.evaluation(model, X_test, y_test)

# Start an MLflow run
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("Training Info", "Decision Tree model for digits_model data")
    
    signature = infer_signature(X_train, model.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train,
        registered_model_name="504cb42f" # Same name as the regresion model
    )
