import pandas as pd
'''
Calculate the the Pearson correlation coefficient against a target variable
and prints the values. This allows us to select features with a strong correlation.
'''


df = pd.read_csv("raw_data.csv")

# Target variable we want to find correlation with
target = 'AQI'

# Calculate the correlation of each feature excluding the target
correlations = df.corr()[target].drop(target)

# Sort features by correlation with the target
correlations_sorted = correlations.abs().sort_values(ascending=False)

# Display the correlations
print(correlations_sorted)
