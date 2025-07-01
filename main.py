import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

class IncomeClassifierPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.scaled_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clf = None
        self.grid_search = None

    def load_data(self):
        self.dataset = pd.read_csv(self.data_path, delimiter=';')
        print("First 5 rows:\n", self.dataset.head())

    def explore_data(self):
        print("\nDataset Count:\n", self.dataset.count())
        print("\nMissing Values Count:\n", self.dataset.isnull().sum())
        print("\nDuplicated Values Count:\n", self.dataset.duplicated().sum())
        print("\nData Types:\n", self.dataset.dtypes)
        print("\nDataset Description:\n", self.dataset.describe())
        print("\nNon-Numerical Columns Information:\n", self.dataset.select_dtypes(include=['object']).describe())

    def visualize_data(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.dataset.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(self.dataset['Alter'], bins=30, kde=True)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(10, 6))
        self.dataset['Einkommen'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
        plt.title('Income Distribution')
        plt.ylabel('')
        plt.show()

        categorical_columns = ['Abschluss', 'Augenfarbe', 'Einkommen']
        for col in categorical_columns:
            if col in self.dataset.columns:
                print(f"Unique values in '{col}': {self.dataset[col].unique()}")

    def preprocess_data(self):
        # Drop unnecessary columns
        if 'Person Nr.' in self.dataset.columns:
            self.dataset = self.dataset.drop(columns=['Person Nr.'])

        categorical_columns = ['Abschluss', 'Augenfarbe', 'Einkommen']
        num_cols = self.dataset.select_dtypes(include=[np.number]).columns
        cat_cols = [col for col in categorical_columns if col in self.dataset.columns]

        # Impute missing values
        num_imputer = SimpleImputer(strategy='mean')
        self.dataset[num_cols] = num_imputer.fit_transform(self.dataset[num_cols])

        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.dataset[cat_cols] = cat_imputer.fit_transform(self.dataset[cat_cols])

        # Encode categorical features
        ordinal_encoder = OrdinalEncoder()
        self.dataset[cat_cols] = ordinal_encoder.fit_transform(self.dataset[cat_cols])

        # Feature scaling
        scaler = MinMaxScaler()
        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.dataset), columns=self.dataset.columns)

    def feature_selection(self):
        # Remove constant features
        constant_features = [col for col in self.scaled_data.columns if self.scaled_data[col].nunique() == 1]
        features = self.scaled_data.drop(columns=constant_features + ['Einkommen'])
        target = self.scaled_data['Einkommen'].astype(int)

        # Remove highly correlated features
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        features = features.drop(columns=to_drop)

        self.X = features
        self.y = target

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print("Training Features Shape:", self.X_train.shape)
        print("Testing Features Shape:", self.X_test.shape)

    def train_model(self):
        self.clf = DecisionTreeClassifier(random_state=42)
        self.clf.fit(self.X_train, self.y_train)
        y_pred = self.clf.predict(self.X_test)
        print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))

    def visualize_tree(self):
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.clf,
            feature_names=self.X.columns,
            class_names=[str(cls) for cls in np.unique(self.y)],
            filled=True,
            rounded=True
        )
        plt.title("Decision Tree Visualization")
        plt.show()

    def hyperparameter_tuning(self):
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        self.grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters:", self.grid_search.best_params_)
        print("Best Cross-Validation Accuracy:", self.grid_search.best_score_)

def main():
    pipeline = IncomeClassifierPipeline('personen_datensatz.csv')
    pipeline.load_data()
    pipeline.explore_data()
    pipeline.visualize_data()
    pipeline.preprocess_data()
    pipeline.feature_selection()
    pipeline.split_data()
    pipeline.train_model()
    pipeline.visualize_tree()
    pipeline.hyperparameter_tuning()

if __name__ == "__main__":
    main()