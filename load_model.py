import mlflow
import numpy as np
import pandas as pd

# TODO: Set tht MLFlow server uri
uri = "http://127.0.0.1:6001"
mlflow.set_tracking_uri(uri)

# TODO: Provide model path/url

# logged_model = "runs:/1e0e976d/digits_model"
# logged_model = "mlflow-artifacts:/718091433743668126/fcc1f21ea7204882bf5c31f5812fd1d9/artifacts/model.pkl"
logged_model = 'runs:/959e316b21124e389760a100deacf7de/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
np.random.seed(42)
data = np.random.rand(1, 64)

# TODO: Predict the output for the data. You might need to use a pandas DataFrame due to a constraint from MLFlow.
prediction = loaded_model.predict(pd.DataFrame(data))

# Print out prediction result
print("La prediccion es: ")
print(prediction)
