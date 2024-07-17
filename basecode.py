import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the training data
train_data = pd.read_csv('/content/train.csv')

# Load the testing data (without SalePrice column)
test_data = pd.read_csv('/content/test.csv')

# Check for missing values in training and testing data
print("Training Data Missing Values:")
print(train_data.isnull().sum())
print("\nTesting Data Missing Values:")
print(test_data.isnull().sum())

# Drop rows with missing values in both training and testing data
train_data_cleaned = train_data.dropna()
test_data_cleaned = test_data.dropna()

# Check if there are any rows left after dropping missing values
print(f"Number of rows in cleaned training data: {train_data_cleaned.shape[0]}")
print(f"Number of rows in cleaned testing data: {test_data_cleaned.shape[0]}")

# If no rows are left, switch to imputation strategy
if train_data_cleaned.shape[0] == 0 or test_data_cleaned.shape[0] == 0:
    from sklearn.impute import SimpleImputer
    print("Too many missing values, switching to imputation.")

    # Define features and target variable for training data
    X_train = train_data[['FullBath', 'HalfBath', 'BedroomAbvGr', 'TotalBsmtSF', '2ndFlrSF', '1stFlrSF']]
    y_train = train_data['SalePrice']

    # Define features for testing data
    X_test = test_data[['FullBath', 'HalfBath', 'BedroomAbvGr', 'TotalBsmtSF', '2ndFlrSF', '1stFlrSF']]

    # Impute missing values using the mean strategy
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Create the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train_imputed, y_train)

    # Make predictions on the test set
    predicted_prices = model.predict(X_test_imputed)
else:
    # Proceed with the cleaned data
    print("Proceeding with dropped missing values.")

    # Define features and target variable for training data
    X_train = train_data_cleaned[['FullBath', 'HalfBath', 'BedroomAbvGr', 'TotalBsmtSF', '2ndFlrSF', '1stFlrSF']]
    y_train = train_data_cleaned['SalePrice']

    # Define features for testing data
    X_test = test_data_cleaned[['FullBath', 'HalfBath', 'BedroomAbvGr', 'TotalBsmtSF', '2ndFlrSF', '1stFlrSF']]

    # Create the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predicted_prices = model.predict(X_test)

# Print only the predicted prices
for price in predicted_prices:
    print(price)
