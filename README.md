# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Algorithm — Decision Tree Classifier for Employee Churn
1. **Import libraries** (pandas, sklearn, matplotlib, etc.).  
2. **Load dataset** (CSV) into a DataFrame.  
3. **Inspect data**: head(), info(), isnull().sum(), target distribution.  
4. **Handle missing values** (drop or impute) and encode categorical variables (LabelEncoder).  
5. **Select features (X)** and **target (y)**.  
6. **Split** data into train and test sets (e.g., 80/20, set random_state for reproducibility).  
7. **Initialize** DecisionTreeClassifier (criterion = "entropy" or "gini").  
8. **Train** model with .fit(X_train, y_train).  
9. **Predict** on X_test and optionally on new samples.  
10. **Evaluate** using accuracy_score, confusion_matrix, classification_report.  
11. **Visualize** the trained tree with plot_tree and save/show the plot.  
12. (Optional) **Tune** hyperparameters or perform cross-validation for better performance.
"""))

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SANTHOSH S
RegisterNumber: 212224100052
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
# Replace the path below with the correct path for your environment if needed.
csv_path = "Employee.csv"   # <-- change if file is somewhere else
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at {csv_path}. Update 'csv_path' to the correct location.")

# 2) Quick inspection
print("Data shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing values per column:\n", data.isnull().sum())
print("\nTarget distribution (left):")
display(data["left"].value_counts())

# 3) Handle missing values (simple approach: drop rows with NA)
if data.isnull().any().any():
    print("\nDropping rows with missing values...")
    data = data.dropna()
    print("New shape after dropna:", data.shape)

# 4) Encode categorical columns
# We encode 'salary'. If you have other categorical columns (e.g., 'department'), consider encoding or one-hot.
if 'salary' in data.columns:
    le = LabelEncoder()
    data['salary'] = le.fit_transform(data['salary'].astype(str))
    print("\nSalary classes (label encoding mapping):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

# 5) Define features (X) and target (y)
feature_cols = [
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company",
    "Work_accident", "promotion_last_5years", "salary"
]

# Validate that all feature columns exist
missing_feats = [c for c in feature_cols if c not in data.columns]
if missing_feats:
    raise ValueError(f"Missing expected feature columns in dataset: {missing_feats}")

X = data[feature_cols]
y = data["left"]

display(X.head())

# 6) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=100, stratify=y if len(y.unique())>1 else None
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# 7) Initialize and train Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

# 8) Predict on test data
y_pred = dt.predict(X_test)

# 9) Evaluation
acc = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc:.4f}")

print("\nConfusion Matrix:")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, digits=4))

# 10) Example: predict for a new employee
# Format: [satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary]
# NOTE: salary must use the same encoding as used above (see mapping printed earlier)
new_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]  # example values
pred = dt.predict(new_employee)
print(f"\nPrediction for new employee {new_employee[0]}: {pred} ->", "left" if pred[0]==1 else "stayed")

# 11) Visualize the decision tree
plt.figure(figsize=(14,10))
plot_tree(
    dt,
    feature_names=feature_cols,
    class_names=['stayed','left'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Employee Churn")
plt.show()

# Optionally save the tree figure
# plt.savefig("employee_churn_tree.png", bbox_inches="tight", dpi=200)

# 12) (Optional) show feature importances
importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
display(importances)
```

## Output:
<img width="1572" height="280" alt="Screenshot 2026-11-25 133654" src="https://github.com/user-attachments/assets/d7182d30-af96-48f1-918d-5999a604e130" />
<img width="506" height="421" alt="Screenshot 2026-11-25 133737" src="https://github.com/user-attachments/assets/ab592b91-f551-4413-80c1-dc551719342a" />
<img width="327" height="313" alt="Screenshot 2026-11-25 133806" src="https://github.com/user-attachments/assets/3a5f63cb-de70-48da-bb55-31a62c059b5f" />
<img width="282" height="230" alt="Screenshot 2026-11-25 133817" src="https://github.com/user-attachments/assets/5ceb7d24-4862-4801-b693-8013e4b00d95" />
<img width="1462" height="305" alt="Screenshot 2026-11-25 133831" src="https://github.com/user-attachments/assets/44add5a9-eeb6-42c8-80cf-0618d7e604ef" />
<img width="760" height="440" alt="Screenshot 2026-11-25 133840" src="https://github.com/user-attachments/assets/597644dc-795b-485c-84f9-dfd766947ddc" />
<img width="1300" height="828" alt="IMG1" src="https://github.com/user-attachments/assets/794a9c3c-a7e3-4484-b619-34171f5ca35e" />
<img width="347" height="423" alt="Screenshot 2026-11-25 133850" src="https://github.com/user-attachments/assets/15da42a6-5313-48b3-8c12-949f597a5974" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
