
# Predicting Critical Temperature of Superconductors With a Linear Regression Model Using L1 (LASSO) & L2 (RIDGE) Regularization Techniques

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import ttest_ind

## Loading the Data
# file path
file_path1 = "train.csv" # previously cleaned
file_path2 = "unique_m.csv" # previously cleaned

# Load the dataset
train = pd.read_csv(file_path1)
unique = pd.read_csv(file_path2)

# Set the maximum number of columns to display to None
pd.set_option('display.max_columns', None)
train.head()
train.dtypes
train.shape
unique.head()
unique.dtypes
unique.shape

## Data Preperation
### Data Scaling
# Standardizing the features using StandardScaler
scaler = StandardScaler()
# Standardizing the features using StandardScaler
# transformer = QuantileTransformer(output_distribution='uniform')

### Seperate Target
# Prepare training data for models
X = unique.drop(['critical_temp', 'material'], axis=1) 
y = unique['critical_temp']

### Train/Test Split
# Splitting the data into training and test sets using 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Model Preperation
# L1 (LASSO)
# Initializing and fitting the Lasso regression model
alpha = 1.0  # Regularization strength
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_train_scaled, y_train)

# L2 (RIDGE)
# Initializing and fitting the Ridge regression model
alpha = 1.0  # Regularization strength
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train_scaled, y_train)

## Model Evaluation
# L1 (LASSO)
# Making predictions on the test set
y_pred_lasso = lasso_model.predict(X_test_scaled)
# Evaluating the Lasso model performance (MSE)
mseL = mean_squared_error(y_test, y_pred_lasso)

# Print the evaluation metrics
print(f"MSE: {mseL}")

# L2 (RIDGE)
# Making predictions on the test set
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluating the Lasso model performance (MSE)
mseR = mean_squared_error(y_test, y_pred_ridge)

# Print the evaluation metrics
print(f"MSE: {mseR}")

## Discuss Important Features
# L1 (LASSO)
# Get the names of the features with non-zero coefficients
selected_features1 = X.columns[lasso_model.coef_ != 0]
print(selected_features1.shape)

# Print the selected features
print(f"Selected Features: {selected_features1}")
print(f'Coefficients with Lasso: {lasso_model.coef_}')
print(f'Intercept with Lasso: {lasso_model.intercept_}')

# L2 (RIDGE)
# Get the names of the features with non-zero coefficients
selected_features2 = X.columns[ridge_model.coef_ != 0]
print(selected_features2.shape)

# Print the selected features
print(f"Selected Features: {selected_features2}")
print(f'Coefficients with Ridge: {ridge_model.coef_}')
print(f'Intercept with Ridge: {ridge_model.intercept_}')

# Find the common features
common_features = set(selected_features1) & set(selected_features2)

# Print the common features
print(len(common_features))
common_features
