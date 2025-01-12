

---

# Electric Vehicle Population Analysis

## 📊 Dataset Overview

This repository contains an analysis of the **Electric Vehicle (EV) Population** across various U.S. states, focusing on trends, pricing, and eligibility for clean alternative fuel vehicle programs. The dataset offers a detailed breakdown of electric vehicles registered in different regions, with key metrics such as electric range, manufacturer information, and legislative eligibility.

### 🌍 **Dataset Description**

The dataset provides the following key columns:

- **State**: The U.S. state where the vehicle is registered.
- **Model Year**: The year the vehicle model was manufactured.
- **Make**: The manufacturer of the vehicle (e.g., TESLA, BMW, CHEVROLET, etc.).
- **Electric Vehicle Type**: Type of electric vehicle, categorized as:
  - Battery Electric Vehicle (BEV)
  - Plug-in Hybrid Electric Vehicle (PHEV)
- **Electric Range**: The range of the vehicle on electric power alone (in miles).
- **Base MSRP**: Manufacturer's suggested retail price (MSRP) of the vehicle.
- **Legislative District**: The legislative district associated with the vehicle’s registration.
- **CAFV Eligibility Simple**: Indicates if the vehicle is eligible for the Clean Alternative Fuel Vehicle (CAFV) program. Possible values:
  - Clean Alternative Fuel Vehicle Eligible
  - Eligibility unknown due to unresearched battery range
  - Not eligible due to low battery range

### 🔍 **Use Cases**

- **Geographic Analysis**: Understand the adoption of EVs in different U.S. states.
- **Pricing Trends**: Analyze the relationship between pricing and electric range.
- **Legislative Impact**: Study how legislation affects the adoption of clean vehicles.
- **Predictive Modeling**: Build models to predict future EV trends based on historical data and legislative factors.

---

## 🚀 **Workflow**

This project involves multiple stages, from data loading to feature engineering, data visualization, and building predictive models. The workflow covers the following steps:

### 1. **Data Loading & Exploration**
- Load the dataset and explore the first few rows to understand its structure.
- Check for missing values, duplicates, and basic statistics.

### 2. **Data Preprocessing**
- Handle missing values by imputing or removing rows/columns as necessary.
- Remove duplicates to ensure data integrity.
- Perform label encoding for categorical variables.

### 3. **Data Visualization**
- Univariate and multivariate analysis using:
  - **Histogram**, **Boxplot**, **Violinplot**, **Countplot**, **Barplot** for univariate distribution.
  - **Pairplot**, **Heatmap**, **Scatterplot**, **Lineplot** for multivariate relationships.
  
### 4. **Feature Engineering**
- Extract useful features like year-over-year growth in EV registrations or pricing trends.
- Engineer new features from existing columns for better model performance.

### 5. **Model Building**
- Implement and evaluate several classification models, such as Logistic Regression, Random Forest, and XGBoost.
- Perform cross-validation to assess model performance.
- Compare the performance of multiple models using metrics like accuracy, precision, recall, and F1-score.

### 6. **Model Evaluation**
- Use metrics like **Confusion Matrix**, **ROC-AUC**, and **Classification Report** to evaluate model performance.
- Visualize the confusion matrix to understand misclassifications.

---

## 📦 **Installation**

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/ev-population-analysis.git
cd ev-population-analysis
pip install -r requirements.txt
```

The `requirements.txt` file includes:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost
- plotly

---

## 📝 **Code Implementation**

Below is an example of how you can use this repository for data analysis and machine learning model building:

### **Data Loading and Preprocessing**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
df = pd.read_csv("/kaggle/input/electric-vehicle-population/EV_Population.csv")

```

### **Data Visualization**

```python
# Plotting the distribution of duplicate values
sns.countplot(data=df, x='Is_Duplicate', palette='coolwarm')
plt.title('Duplicate Rows Count')
plt.xlabel('Is Duplicate')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Unique', 'Duplicate'])
plt.show()
duplicate_rows = df[df['Is_Duplicate']]

plt.figure(figsize=(12, 6))
sns.heatmap(duplicate_rows.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Duplicate Rows')
plt.show()

# Remove duplicate rows and keep the first occurrence
df = df.drop_duplicates(keep='first')

# Reset the index (optional)
df.reset_index(drop=True, inplace=True)

print("Dataset after removing duplicates:", df)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Make", palette="viridis")
plt.title("Count of Vehicles by Make")
plt.xlabel("Make")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Electric Range", kde=True, bins=10, color="blue")
plt.title("Electric Range Distribution")
plt.xlabel("Electric Range (miles)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
corr_matrix = df.corr(numeric_only=True)  # Compute correlations for numeric columns
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

### **Label Encoding for Categorical Features**

```python
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Dataset after Label Encoding:")
print(df)
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Identify numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Apply MinMaxScaler to numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("Dataset after Min-Max Scaling:")
print(df)
# Identify numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create boxplots for each numerical column
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 1, i)
    sns.boxplot(data=df, x=col, color='skyblue')
    plt.title(f'Boxplot for {col}', fontsize=14)
    plt.xlabel('')
    plt.tight_layout()

plt.show()
# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile
    Q3 = df[column].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1                   # Interquartile range
    
    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter rows within bounds
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Identify numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Remove outliers from all numerical columns
for col in numerical_columns:
    df = remove_outliers_iqr(df, col)

print("Dataset after removing outliers:")
print(df)
```

### **Model Building and Evaluation**

```python


# Split the data into features and target variable
X = df.drop('Electric Vehicle Type', axis=1)  # Replace 'Target' with the actual target column name
y = df['Electric Vehicle Type']  # Replace 'Target' with the actual target column name
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
# Define a list of classifiers
classifiers = [
    ('Logistic Regression', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('SVM', SVC(probability=True)),
    ('Dummy Classifier', DummyClassifier(strategy='most_frequent')),
    ('Voting Classifier', VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True))
    ], voting='soft'))
]
# Initialize results
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'ROC AUC': []
}
# Apply each classifier and evaluate performance
for name, clf in classifiers:
    # Cross-validation score
    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
from sklearn.preprocessing import LabelBinarizer

# Check if the problem is binary or multi-class
is_binary = len(y.unique()) == 2  # Check if it's a binary classification problem

# Evaluate the models
for name, clf in classifiers:
    # Cross-validation score
    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC AUC score for multi-class (One-vs-Rest)
    if not is_binary:  # Only compute ROC AUC for multi-class problems
        lb = LabelBinarizer()
        lb.fit(y)
        y_test_bin = lb.transform(y_test)
        y_pred_prob = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr', average='weighted')
    else:  # For binary classification, use the original approach
        y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
        roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)
    results['ROC AUC'].append(roc_auc)

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# Display the results
print(results_df)

# Plot the comparison of models' performance using accuracy, precision, recall, and F1 Score
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy Plot
sns.barplot(x='Accuracy', y='Model', data=results_df, ax=ax[0, 0])
ax[0, 0].set_title('Accuracy Comparison')

# Precision Plot
sns.barplot(x='Precision', y='Model', data=results_df, ax=ax[0, 1])
ax[0, 1].set_title('Precision Comparison')

# Recall Plot
sns.barplot(x='Recall', y='Model', data=results_df, ax=ax[1, 0])
ax[1, 0].set_title('Recall Comparison')

# F1 Score Plot
sns.barplot(x='F1 Score', y='Model', data=results_df, ax=ax[1, 1])
ax[1, 1].set_title('F1 Score Comparison')

plt.tight_layout()
plt.show()


```
![Alt Text](https://github.com/Arif-miad/Electric-Vehicle-Population-Analysis/blob/main/accuracy.PNG)
![Alt Text](https://github.com/Arif-miad/Electric-Vehicle-Population-Analysis/blob/main/plots.PNG)


---

## 🔍 **Future Work**

- **Hyperparameter Tuning**: Implement grid search or randomized search to tune model parameters for better accuracy.
- **Time-Series Analysis**: Investigate the impact of time (model year) on EV adoption and pricing.
- **Predictive Modeling**: Build advanced predictive models to forecast future EV trends across states.

---

