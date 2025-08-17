
# === 1. IMPORTS & SETUP ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Evaluation
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# Explainability
import shap

# --- Define Client Brand Colors (from Phase 1) ---
NO_CHURN_COLOR = '#FF004F'
CHURN_COLOR = '#00FFBF'
custom_palette = {1: CHURN_COLOR, 0: NO_CHURN_COLOR}
sns.set(style="whitegrid")


# === 2. DATA LOADING & PREPARATION ===
# Load data (assuming the CSV is in the same directory)
df = pd.read_csv('C:\\Users\\esteb\\Documents\\GitHub\\customer-churn-and-personalisation\\data\\Vodafone_Customer_Churn_Sample_Dataset.csv')

# Data Cleaning (as in Phase 1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df.drop('customerID', axis=1, inplace=True)

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()


# === 3. PREPROCESSING PIPELINE ===
# Create preprocessing steps for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any), though we've handled all
)

# Split data into training and testing sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# === 4. MODEL TRAINING ===

# --- Model 1: Logistic Regression (Baseline) ---
# Create the full pipeline including preprocessing and the model
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))])
print("Training Logistic Regression model...")
lr_pipeline.fit(X_train, y_train)
print("Done.\n")


# --- Model 2: XGBoost (Industry Standard) ---
# Calculate scale_pos_weight for XGBoost to handle class imbalance
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, 
                                                            eval_metric='logloss', scale_pos_weight=scale_pos_weight))])
print("Training XGBoost model...")
xgb_pipeline.fit(X_train, y_train)
print("Done.\n")


# --- Model 3: Neural Network (Academic High-Performer) ---
# We need to preprocess the data first to get the input shape for the network
X_train_processed = lr_pipeline.named_steps['preprocessor'].transform(X_train)
X_test_processed = lr_pipeline.named_steps['preprocessor'].transform(X_test)
input_shape = X_train_processed.shape[1]

# Define the Keras model
def create_nn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

nn_model = create_nn_model(input_shape)
print("Neural Network Model Summary:")
nn_model.summary()

# Calculate class weights for Keras
class_weight_0 = (1 / y_train.value_counts()[0]) * (len(y_train) / 2.0)
class_weight_1 = (1 / y_train.value_counts()[1]) * (len(y_train) / 2.0)
nn_class_weights = {0: class_weight_0, 1: class_weight_1}

print("\nTraining Neural Network model...")
history = nn_model.fit(X_train_processed, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_split=0.2,
                       class_weight=nn_class_weights,
                       verbose=0) # verbose=0 suppresses epoch-by-epoch output for cleaner results
print("Done.\n")


# === 5. MODEL EVALUATION ===

# Get predictions
lr_preds = lr_pipeline.predict(X_test)
xgb_preds = xgb_pipeline.predict(X_test)
nn_preds_proba = nn_model.predict(X_test_processed).flatten()
nn_preds = (nn_preds_proba > 0.5).astype(int) # Convert probabilities to 0/1 predictions

# Get prediction probabilities for AUC scores
lr_preds_proba = lr_pipeline.predict_proba(X_test)[:, 1]
xgb_preds_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

# --- Print Classification Reports ---
print("="*60)
print("                Model Evaluation Results")
print("="*60)

print("\n--- Logistic Regression ---")
print(classification_report(y_test, lr_preds, target_names=['No Churn (0)', 'Churn (1)']))
print(f"AUC-ROC Score: {roc_auc_score(y_test, lr_preds_proba):.4f}")

print("\n--- XGBoost Classifier ---")
print(classification_report(y_test, xgb_preds, target_names=['No Churn (0)', 'Churn (1)']))
print(f"AUC-ROC Score: {roc_auc_score(y_test, xgb_preds_proba):.4f}")

print("\n--- Neural Network ---")
print(classification_report(y_test, nn_preds, target_names=['No Churn (0)', 'Churn (1)']))
print(f"AUC-ROC Score: {roc_auc_score(y_test, nn_preds_proba):.4f}")
print("="*60)