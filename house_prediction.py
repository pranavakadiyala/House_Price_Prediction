
#In this document I am only training the best performing model - which was the lasso model

import mlflow
from mlflow.models import infer_signature


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import io




def generate_plot_and_metrics():
    # Use the 'Agg' backend for Matplotlib
    matplotlib.use('Agg')
    sns.set()  # Set Seaborn style for the plot


    # Enable automatic logging to MLflow
    mlflow.sklearn.autolog()

    #read data
    df = pd.read_csv("Housing.csv")
    object_columns = df.select_dtypes(include=['object']).columns
    cont_columns = df.select_dtypes(include=['int64']).columns

    #feature engineering
    seed = 13
    #encoding categorical, non-numerical variables
    pd_dummy = pd.get_dummies(data=df[object_columns])
    df = pd.concat([df,pd_dummy],axis=1).drop(columns=object_columns)

    X = df[df.columns.difference(['price'])]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)


    # Start MLflow run
    with mlflow.start_run():
        # Train a regression model
        params_lasso = {"alpha": 1e-6, "normalize":True, "random_state":seed}
        model = Lasso(**params_lasso)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        #r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {'MAE': mae}

        # Log metrics manually (optional, since autolog will capture them)
        mlflow.log_metric('MSE', mse)
        #mlflow.log_metric('R2', r2)
        mlflow.log_metric('MAE', mae)

        # Log model manually (optional, since autolog will capture it)
        mlflow.sklearn.log_model(model, "lasso-regression-model")

        # Generate a plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_test[:, 0], y=y_test, color='green', label='Actual', s=50)  # s is the size of the dots
        sns.lineplot(x=X_test[:, 0], y=y_pred, color='orange', label='Predicted', linewidth=2)
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Regression Plot')
        plt.legend()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

    return buf, metrics


