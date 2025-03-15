import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load dataset
data = pd.read_csv("Heart Attack.csv")

# Rename columns and create new features
data.rename(columns={'impluse': 'pulse', 'pressurehight': 'sys', 'pressurelow': 'dias'}, inplace=True)
data['glu_mmol'] = data['glucose'] / 18

# Recoding categorical variables
data['gender'] = data['gender'].map({1: 'Male', 0: 'Female'}).astype('category')
data['class'] = data['class'].astype('category')

# Select relevant features
features = ['age', 'gender', 'pulse', 'sys', 'dias', 'kcm', 'glu_mmol', 'troponin']
target = 'class'
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=125)

# Preprocessing pipeline
numeric_features = ['age', 'pulse', 'sys', 'dias', 'kcm', 'glu_mmol', 'troponin']
categorical_features = ['gender']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Build logistic regression model
log_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Cross-validation
scores = cross_val_score(log_model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy: {scores.mean():.3f}')

# Fit model
log_model.fit(X_train, y_train)

# Predictions
y_pred = log_model.predict(X_test)
y_pred_prob = log_model.predict_proba(X_test)[:, 1]

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test.cat.codes, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

