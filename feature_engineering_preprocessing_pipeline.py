"""Feature Engineering & Preprocessing Pipeline - exported notebook code."""
# Install: pip install pandas numpy matplotlib seaborn scikit-learn joblib

# If needed, uncomment in Jupyter/Colab:
# %pip install -U pandas numpy matplotlib seaborn scikit-learn joblib

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

print("scikit-learn version:", sklearn.__version__)

# Put customer-churn-training.csv in the same folder as this notebook.
# In Google Colab, upload the CSV or change DATA_PATH to its location.
DATA_PATH = Path("customer-churn-training.csv")
TARGET = "churned"
ID_COLUMNS = ["customer_id"]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Upload the CSV or update DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
display(df.head())
display(df.info())

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' is missing.")

if df[TARGET].isna().any():
    raise ValueError("Target contains missing values. Resolve labels before training.")

# Remove exact duplicate rows; preserve original data in the source file.
df = df.drop_duplicates().copy()

# This notebook is a binary classification example.
target_values = set(df[TARGET].unique())
if not target_values.issubset({0, 1}):
    raise ValueError(f"Expected binary target labels 0/1, found {target_values}")

df[TARGET] = df[TARGET].astype(int)

print("Missing values by column:")
display(df.isna().sum().to_frame("missing_count"))

print("Duplicate rows remaining:", df.duplicated().sum())
print("Target distribution:")
display(df[TARGET].value_counts().rename("count").to_frame())

# IDs are identifiers, not predictive features.
columns_to_drop = [TARGET] + [c for c in ID_COLUMNS if c in df.columns]
X = df.drop(columns=columns_to_drop).copy()
y = df[TARGET].copy()

# Detect columns from their data types so the notebook is reusable.
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Feature matrix shape:", X.shape)

class_counts = y.value_counts()
if len(class_counts) < 2 or class_counts.min() < 2:
    raise ValueError(
        "Need at least two examples from each class for this stratified split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train target distribution:")
display(y_train.value_counts(normalize=True).rename("proportion").to_frame())
print("Test target distribution:")
display(y_test.value_counts(normalize=True).rename("proportion").to_frame())

# Numerical: median imputation followed by standardization.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: most-frequent imputation followed by one-hot encoding.
# handle_unknown='ignore' prevents errors for unseen test/deployment categories.
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

print("Preprocessing transformer created.")

# The ColumnTransformer is fitted only when pipeline.fit(X_train, y_train) runs.
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

model_pipeline.fit(X_train, y_train)
print("Pipeline training complete.")

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": (
        roc_auc_score(y_test, y_prob)
        if y_test.nunique() == 2 else np.nan
    )
}

display(pd.DataFrame([metrics]).T.rename(columns={0: "Test score"}))
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Retained", "Churned"],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Retained", "Churned"],
    yticklabels=["Retained", "Churned"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

if numeric_features:
    train_numeric = X_train[numeric_features].copy()
    train_numeric = train_numeric.apply(
        lambda col: col.fillna(col.median())
    )

    corr = train_numeric.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Training Data: Numerical Feature Correlation")
    plt.tight_layout()
    plt.show()
else:
    print("No numerical predictors available for correlation analysis.")

# Transform using the already-fitted training-only preprocessor.
X_train_transformed = model_pipeline.named_steps["preprocessor"].transform(X_train)

# Obtain transformed feature names (including one-hot encoded categories).
feature_names = model_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# Mutual information supports sparse matrices; convert to dense for this small demo.
if hasattr(X_train_transformed, "toarray"):
    X_train_dense = X_train_transformed.toarray()
else:
    X_train_dense = np.asarray(X_train_transformed)

mi_scores = mutual_info_classif(
    X_train_dense,
    y_train,
    discrete_features="auto",
    random_state=RANDOM_STATE
)

mi_ranking = pd.DataFrame({
    "feature": feature_names,
    "mutual_information": mi_scores
}).sort_values("mutual_information", ascending=False)

display(mi_ranking)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=mi_ranking,
    x="mutual_information",
    y="feature"
)
plt.title("Feature Ranking - Mutual Information (Training Set)")
plt.xlabel("Mutual information")
plt.ylabel("Transformed feature")
plt.tight_layout()
plt.show()

rf = model_pipeline.named_steps["classifier"]

tree_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

display(tree_importance)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=tree_importance,
    x="importance",
    y="feature"
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Impurity-based importance")
plt.ylabel("Transformed feature")
plt.tight_layout()
plt.show()

MODEL_PATH = "customer_churn_preprocessing_pipeline.joblib"
joblib.dump(model_pipeline, MODEL_PATH)
print("Saved:", MODEL_PATH)

# Example: load and predict using raw feature columns.
loaded_pipeline = joblib.load(MODEL_PATH)

example_customer = pd.DataFrame([{
    "tenure_months": 5,
    "support_tickets": 3,
    "monthly_spend_inr": 799,
    "last_login_days": 15,
    "plan_type": "Basic"
}])

example_customer = example_customer.reindex(columns=X.columns)
prediction = loaded_pipeline.predict(example_customer)[0]
probability = loaded_pipeline.predict_proba(example_customer)[0, 1]

print("Predicted class:", "Churned" if prediction == 1 else "Retained")
print(f"Estimated churn probability: {probability:.3f}")

