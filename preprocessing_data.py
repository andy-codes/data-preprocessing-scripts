import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
Script to prepare a pre-processed dataset split into training data and test data. This is based on the
raw dataset: Air Quality Data in India (2015-2020).

The script prepares two output files in CSV format:

1. 'train_dataset_scaled.csv': Scaled training dataset.
2.  'test_dataset_scaled.csv': Scaled testing dataset.


The script is a basic sequential set of actions that outputs intermediary datasets allowing for 
inspect of the data as different operations are run. 

Broadly the script:

1. Identifies and removes rows where all values in selected columns are either NaN or zero.
2. Drops specific columns that have low correlation with the target variable.
3. Uses an iterative imputation to estimate and fill in missing values based on Bayesian Ridge regression.
4. Scales the feature data to standardise the range of values for training and testing datasets.
5. Writes the cleaned, imputed, and scaled datasets to separate CSV files for training and testing.

"""

# ===========================
# Remove Empty Rows
# ===========================

df = pd.read_csv('raw_data.csv')

# Define the columns that must be checked for missing values (everything except "City" and "Date")
columns_to_check = df.columns.difference(['City', 'Date'])

# Replace any empty strings with NaN to standardize missing values
df[columns_to_check] = df[columns_to_check].replace('', np.nan)

# Create a mask that checks where all columns_to_check are either NaN or 0
mask = (df[columns_to_check].isna() | (df[columns_to_check] == 0)).all(axis=1)

# Drop rows where all columns_to_check are NaN or 0
df_cleaned = df[~mask]

stage_output_file = '1_empty_rows_removed.csv'

df_cleaned.to_csv(stage_output_file, index=False)


# ==============================
# Remove low correlation columns
# ===============================

df = pd.read_csv(stage_output_file)

# Drop columns that we do NOT want in the dataset
df = df.drop(['City', 'Date', 'Toluene', 'NH3', 'O3', 'Xylene', 'Benzene', 'AQI_Bucket', 'PM10'], axis=1)

stage_output_file = '2_low_correlation_removed.csv'

df.to_csv(stage_output_file, index=False)


# ==============================
# Impute missing data and generate final preprocessed data sets
# ===============================

df = pd.read_csv(stage_output_file)

# Separate features and target (AQI column). We don't want to impute AQI as it's the target variable.
features = df.drop(columns=['AQI'])
target = df['AQI']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Impute missing values with IterativeImputer
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=200, random_state=42)

# Fit the imputer on the training data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create scaler
scaler = StandardScaler()
scaler_target = StandardScaler()

# Fit the scaler on the imputed training data and target
X_train_scaled = scaler.fit_transform(X_train_imputed)
y_train_scaled = scaler_target.fit_transform(y_train.values.reshape(-1, 1))

# Apply the scaling to the imputed test data and target
X_test_scaled = scaler.transform(X_test_imputed)
y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1))

# Recombine features and scaled target for both training and test sets
train_data = pd.DataFrame(X_train_scaled, columns=features.columns)
train_data['AQI_scaled'] = y_train_scaled.flatten()

test_data = pd.DataFrame(X_test_scaled, columns=features.columns)
test_data['AQI_scaled'] = y_test_scaled.flatten()

# Create final data sets
train_data.to_csv('train_dataset_scaled.csv', index=False)
test_data.to_csv('test_dataset_scaled.csv', index=False)

