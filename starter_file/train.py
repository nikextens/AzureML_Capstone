from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):
    # Dict for cleaning data
    
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df["type"] = x_df.type.apply(lambda s: 1 if s == "white" else 0)

    y_df = x_df.pop("quality")
    return x_df, y_df

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"


url_path='https://docs.google.com/spreadsheets/d/e/2PACX-1vQ0_ymTF3299kfZvr0KJq5JMLX7ZK4yRg9RXYTqEsMqm2eeUrABv_4MVQMzrqfw1CsbmcrnTqIluMA0/pub?output=csv'
ds = TabularDatasetFactory.from_delimited_files(path=url_path)
print(ds.to_pandas_dataframe().head())
x, y = clean_data(ds)


# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# ## YOUR CODE HERE ###a

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()


