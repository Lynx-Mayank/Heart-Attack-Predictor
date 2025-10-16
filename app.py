"""
Heart Attack Prediction - Colab-ready Python script (with interactive prediction)

Usage:
 - Run this directly in Google Colab.
 - Upload your Excel dataset when prompted.
 - Automatically detects dataset, trains multiple models, evaluates them, and saves the best one.
 - At the end, you can enter new patient data interactively to predict heart attack risk.

Output:
 - Displays model comparison and ROC curve.
 - Saves the best trained model as /content/heart_attack_model.pkl.

"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import joblib
from google.colab import files

# ------------------ Step 1: Upload dataset ------------------
print("Please upload your Heart Attack Prediction Dataset (.xlsx)...")
uploaded = files.upload()
DATA_PATH = Path(list(uploaded.keys())[0])
print(f"Loaded file: {DATA_PATH}")

# ------------------ Step 2: Load and preprocess ------------------

xls = pd.ExcelFile(DATA_PATH)
df = pd.read_excel(xls, sheet_name=0)
print(f"Loaded sheet: {xls.sheet_names[0]}  — shape: {df.shape}")

n = len(df)
unique_counts = df.nunique(dropna=False)
id_like = unique_counts[unique_counts >= (0.95 * n)].index.tolist()
if id_like:
    print(f"Dropping likely ID columns: {id_like}")
    df = df.drop(columns=id_like)

target_candidates = ['target', 'heartdisease', 'heart_disease', 'heartattack', 'heart_attack',
                     'chd', 'cardiac_event', 'output', 'label', 'death_event', 'response', 'has_heart_attack', 'heart attack risk']
target_col = None
for cand in target_candidates:
    for c in df.columns:
        if cand in c.lower():
            target_col = c
            break
    if target_col:
        break
if target_col is None:
    target_col = df.columns[-1]
    print(f"No standard target found. Using last column: '{target_col}'")
else:
    print(f"Detected target column: '{target_col}'")

X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

if len(pd.Series(y).dropna().unique()) > 2:
    median_val = pd.Series(y).median()
    print(f"Target has >2 unique values. Binarizing by > median ({median_val}).")
    y = (y > median_val).astype(int)

numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool', 'datetime']).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(str)

constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if constant_cols:
    print(f"Dropping constant columns: {constant_cols}")
    X = X.drop(columns=constant_cols)
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# ------------------ Step 3: Preprocessing ------------------

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')

# ------------------ Step 4: Model Training ------------------

models = {
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}
for name, clf in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([('preproc', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else pipe.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'roc': roc, 'pipe': pipe}
    print(f"→ {name}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, roc_auc={roc:.4f}")

# ------------------ Step 5: Evaluation ------------------

best_name = max(results.keys(), key=lambda k: results[k]['roc'])
best_model = results[best_name]['pipe']
print(f"\nBest Model: {best_name}")

summary = pd.DataFrame([{ 'Model': k, **results[k]} for k in results])
print("\nModel Performance Summary:")
print(summary[['Model', 'acc', 'prec', 'rec', 'f1', 'roc']])

# ------------------ Step 6: Visualize & Save ------------------

y_proba_best = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'{best_name} (AUC={results[best_name]["roc"]:.2f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve — {best_name}')
plt.legend()
plt.show()

cm = confusion_matrix(y_test, best_model.predict(X_test))
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

MODEL_PATH = Path("/content/heart_attack_model.pkl")
joblib.dump(best_model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print("\nTo download the model:")
print("from google.colab import files\nfiles.download('/content/heart_attack_model.pkl')")

# ------------------ Step 7: Interactive Prediction ------------------

print("\n\n🔍 Interactive Prediction: Enter patient data below to test the model.")
new_patient = {}

for col in X.columns:
    val = input(f"Enter value for {col}: ")
    try:
        val = float(val)
    except ValueError:
        val = str(val)
    new_patient[col] = [val]

new_df = pd.DataFrame(new_patient)
prob = best_model.predict_proba(new_df)[:, 1][0]
pred = best_model.predict(new_df)[0]

print("\n--- Prediction Result ---")
print(f"Heart Attack Risk: {'HIGH' if pred == 1 else 'LOW'}")
print(f"Predicted Probability: {prob:.2f}")
