# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

# 2. Load Dataset
dataset = pd.read_csv('personen_datensatz.csv', delimiter=';')
print("First 5 rows:\n", dataset.head())

# 3. Data Exploration
print("\nDataset Count:\n", dataset.count())
print("\nMissing Values Count:\n", dataset.isnull().sum())
print("\nDuplicated Values Count:\n", dataset.duplicated().sum())
print("\nData Types:\n", dataset.dtypes)
print("\nDataset Description:\n", dataset.describe())
print("\nNon-Numerical Columns Information:\n", dataset.select_dtypes(include=['object']).describe())

# 4. Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(dataset['Alter'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
dataset['Einkommen'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
plt.title('Income Distribution')
plt.ylabel('')
plt.show()

categorical_columns = ['Abschluss', 'Augenfarbe', 'Einkommen']
for col in categorical_columns:
    print(f"Unique values in '{col}': {dataset[col].unique()}")

# 5. Data Preprocessing
# Drop unnecessary columns
dataset = dataset.drop(columns=['Person Nr.'])

# Handle missing values
num_cols = dataset.select_dtypes(include=[np.number]).columns
cat_cols = [col for col in categorical_columns if col in dataset.columns]

num_imputer = SimpleImputer(strategy='mean')
dataset[num_cols] = num_imputer.fit_transform(dataset[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
dataset[cat_cols] = cat_imputer.fit_transform(dataset[cat_cols])

# 6. Encoding Categorical Features
ordinal_encoder = OrdinalEncoder()
dataset[cat_cols] = ordinal_encoder.fit_transform(dataset[cat_cols])

# 7. Feature Scaling
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

# 8. Feature Selection
# Remove constant features
constant_features = [col for col in scaled_data.columns if scaled_data[col].nunique() == 1]
X = scaled_data.drop(columns=constant_features + ['Einkommen'])
y = scaled_data['Einkommen'].astype(int)  # Convert target to integer for classification

# Remove highly correlated features (optional)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X = X.drop(columns=to_drop)

# 9. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

# 10. Modeling
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 11. Evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# 12. Visualization of the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in np.unique(y)], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# 13. Hyperparameter Tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)