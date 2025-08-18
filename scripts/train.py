# === 1. IMPORTS & SETUP ===
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === 2. CONFIGURATION ===
# Define file paths and model parameters in one place
class Config:
    DATA_FILE = 'Vodafone_Customer_Churn_Sample_Dataset.csv'
    MODEL_OUTPUT_DIR = 'production_model'
    PREPROCESSOR_FILE = os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.joblib')
    MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, 'churn_model.keras')

# === 3. THE CHURN PREDICTOR CLASS ===
class ChurnPredictor:
    """
    A class to encapsulate the churn prediction model workflow, from training to prediction.
    """
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.numerical_features = []
        self.categorical_features = []

    def _build_preprocessor(self):
        """Builds the scikit-learn preprocessing pipeline."""
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), self.categorical_features)
            ],
            remainder='passthrough'
        )

    def _build_model(self, input_dim):
        """Builds the Keras Neural Network model."""
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'), BatchNormalization(), Dropout(0.4),
            Dense(32, activation='relu'), BatchNormalization(), Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, 
                      loss='binary_crossentropy', 
                      metrics=[tf.keras.metrics.AUC(name='auc')])
        return model

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the preprocessor and the neural network model.
        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target labels.
        """
        print("Starting model training...")
        
        # Identify feature types from the training data
        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = X.select_dtypes(include='object').columns.tolist()
        
        # Build and fit the preprocessor
        self.preprocessor = self._build_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X)
        
        # Build the model
        input_dim = X_train_processed.shape[1]
        self.model = self._build_model(input_dim)
        
        # Calculate class weights for imbalance
        class_weight_0 = (1 / y.value_counts()[0]) * (len(y) / 2.0)
        class_weight_1 = (1 / y.value_counts()[1]) * (len(y) / 2.0)
        nn_class_weights = {0: class_weight_0, 1: class_weight_1}

        # Define callbacks for intelligent training
        early_stopping = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=5, min_lr=1e-5, mode='max')
        
        print("Training Neural Network...")
        self.model.fit(
            X_train_processed, y,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            class_weight=nn_class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1 # Show progress during training
        )
        print("Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates class predictions (0 or 1) for new data.
        Args:
            X (pd.DataFrame): New data for prediction.
        Returns:
            np.ndarray: An array of churn predictions (0 or 1).
        """
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("Model has not been trained or loaded. Please train or load a model first.")
        
        X_processed = self.preprocessor.transform(X)
        predictions = self.predict_proba(X)
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates churn probabilities for new data.
        Args:
            X (pd.DataFrame): New data for prediction.
        Returns:
            np.ndarray: An array of churn probabilities (0.0 to 1.0).
        """
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("Model has not been trained or loaded. Please train or load a model first.")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed).flatten()

    def save_model(self, model_dir: str = Config.MODEL_OUTPUT_DIR):
        """
        Saves the trained preprocessor and model to disk.
        Args:
            model_dir (str): The directory to save the model components.
        """
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("No trained model to save.")
        
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
        self.model.save(os.path.join(model_dir, 'churn_model.keras'))
        print(f"Model saved successfully to directory: {model_dir}")

    @classmethod
    def load_model(cls, model_dir: str = Config.MODEL_OUTPUT_DIR):
        """
        Loads a pre-trained model and preprocessor from disk.
        Args:
            model_dir (str): The directory containing the model components.
        Returns:
            ChurnPredictor: An instance of the class with the loaded model.
        """
        predictor = cls()
        predictor.preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
        predictor.model = load_model(os.path.join(model_dir, 'churn_model.keras'))
        print(f"Model loaded successfully from directory: {model_dir}")
        return predictor

# === 4. MAIN EXECUTION SCRIPT ===
if __name__ == "__main__":
    # --- Step 1: Load and Prepare Data ---
    df = pd.read_csv(Config.DATA_FILE)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.drop('customerID', axis=1, inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Step 2: Train and Save the Model ---
    predictor = ChurnPredictor()
    predictor.train(X_train, y_train)
    predictor.save_model()

    # --- Step 3: Load the Model and Evaluate on Test Data ---
    # This simulates a real-world scenario where you load a pre-trained model for inference.
    print("\n--- Simulating Production Inference ---")
    loaded_predictor = ChurnPredictor.load_model()
    
    # Make predictions on the unseen test set
    test_predictions = loaded_predictor.predict(X_test)
    test_probabilities = loaded_predictor.predict_proba(X_test)
    
    # --- Step 4: Report Final Performance ---
    print("\n--- Final Model Performance on Unseen Test Data ---")
    print(classification_report(y_test, test_predictions, target_names=['No Churn (0)', 'Churn (1)']))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, test_probabilities):.4f}")
    
    # --- Step 5: Example Prediction on a Single Customer ---
    print("\n--- Example: Predicting on a single new customer ---")
    # Create a new customer profile matching the training data columns
    new_customer = pd.DataFrame([{
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
        'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85, 'TotalCharges': 29.85
    }])
    
    churn_probability = loaded_predictor.predict_proba(new_customer)[0]
    print(f"Predicted churn probability for the new customer: {churn_probability:.4f}")
    if churn_probability > 0.5:
        print("Prediction: Customer is AT RISK of churning.")
    else:
        print("Prediction: Customer is likely to stay.")
