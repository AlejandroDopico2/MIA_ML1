import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file into a DataFrame
data = pd.read_csv('Project\dataset\Android_Malware.csv', index_col=False, usecols=lambda column: column != 'Unnamed: 0')
print(data.head())  # Display the first few rows of the test data

# Perform train-test split (keeping stratification based on the 'Label' column)
train_data, test_data = train_test_split(data, test_size=0.01, stratify=data['Label'], random_state=42)

# Display the test data
print(test_data.head())  # Display the first few rows of the test data

print(type(test_data))

test_data.to_csv('Project\dataset\super_simplified_Android_Malware.csv', index = False)