import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('accident_data.csv')  # Replace with the actual path to your dataset

# Display the first few rows and data info
print(data.head())
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Basic statistical summary
print(data.describe())

# Distribution of the target variable
print(data['Accident Severity'].value_counts(normalize=True))

# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Separate features and target
X = data.drop('Accident Severity', axis=1)
y = data['Accident Severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = ['Speed of the vehicle', 'Age', 'Number of lanes', 'Lane width', 'Speed Limit on the road']
categorical_features = ['Gender', 'Vehicle type', 'Road type', 'Alcohol consumption', 'Type of crash', 'Seatbelt usage', 'Road surface condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
    ])

# Create a pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = model.named_steps['classifier'].feature_importances_
feature_names = (numeric_features + 
                 model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names(categorical_features).tolist())

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Function to predict accident severity
def predict_severity(data):
    return model.predict(data)

# Example prediction
example_data = pd.DataFrame({
    'Speed of the vehicle': [50],
    'Age': [30],
    'Gender': ['Male'],
    'Vehicle type': ['Car'],
    'Number of lanes': [2],
    'Lane width': [3.5],
    'Road type': ['Urban'],
    'Alcohol consumption': ['No'],
    'Type of crash': ['Rear-end'],
    'Seatbelt usage': ['Yes'],
    'Speed Limit on the road': [60],
    'Road surface condition': ['Dry']
})

print("Predicted severity:", predict_severity(example_data)[0])
