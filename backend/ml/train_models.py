"""Production-ready multiclass disease prediction training pipeline.

This script loads a cleaned disease dataset, preprocesses features and labels,
trains multiple classification models, evaluates performance, and saves the
best-performing model along with preprocessing artifacts.

Compatible with: Python 3.11+, scikit-learn 1.9+, TensorFlow 2.x, XGBoost 3.x
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_KEYWORDS = ["disease", "diagnosis", "condition", "label", "outcome"]
IGNORED_FEATURE_KEYWORDS = ["target", "label"]
IGNORED_COLUMNS = ["recommended_medicine", "secondary_medicine", "reaction_occurred", "reaction_name", "name"]

# Feature columns for the new dataset
NUMERICAL_FEATURES = [
    "age_exact", "weight_kg", "height_cm", "bmi", "heart_rate", "cholesterol"
]

CATEGORICAL_FEATURES = [
    "gender", "smoking", "alcohol", "exercise_level",
    "diet_quality", "family_history", "bp", "blood_sugar"
]

SYMPTOM_FEATURES = ["symptom_1", "symptom_2", "symptom_3", "symptom_4",
                    "symptom_5", "symptom_6", "symptom_7"]


def default_dataset_path() -> Path:
    """Return the default dataset path relative to this script."""
    return Path(__file__).resolve().parents[2] / "symptom_based_medicine_recommendation_dataset.csv"


def detect_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the target column in the dataset."""
    lower_names = [col.lower() for col in df.columns]
    
    # Try keyword-based detection
    for keyword in TARGET_KEYWORDS:
        if keyword in lower_names:
            target = df.columns[lower_names.index(keyword)]
            logger.info(f"Detected target column by keyword: {target}")
            return target
    
    # Try heuristic: largest categorical column
    object_columns = [col for col in df.columns 
                      if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col])]
    candidates = [col for col in object_columns 
                  if not any(skip in col.lower() for skip in IGNORED_FEATURE_KEYWORDS)]
    
    if candidates:
        target = max(candidates, key=lambda col: df[col].nunique())
        logger.info(f"Detected target column by heuristic: {target}")
        return target
    
    raise ValueError("Unable to automatically detect a target column. Please specify --target.")


def load_dataset(csv_path: Path, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset and separate features from target."""
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if target_column is None:
        target_column = "disease"  # Default for the new dataset
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    y = df[target_column]
    
    # Drop target column and ignored columns
    columns_to_drop = [target_column] + [col for col in IGNORED_COLUMNS if col in df.columns]
    X = df.drop(columns=columns_to_drop)
    
    logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Ignored columns: {', '.join(columns_to_drop)}")
    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, ColumnTransformer]:
    """Preprocess features and labels using ColumnTransformer pipeline."""
    logger.info("Preprocessing data...")
    
    # Encode target labels
    logger.info("Encoding target labels")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create preprocessing pipeline with ColumnTransformer
    logger.info("Creating preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "symptoms",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                SYMPTOM_FEATURES,
            ),
        ],
        remainder="drop",
    )
    
    # Fit and transform the features
    logger.info("Fitting and transforming features...")
    X_processed = preprocessor.fit_transform(X)
    
    # Save preprocessors
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
    logger.info(f"Saved label encoder to {output_dir / 'label_encoder.pkl'}")
    
    joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
    logger.info(f"Saved preprocessor to {output_dir / 'preprocessor.pkl'}")
    
    logger.info(f"Preprocessed dataset: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X_processed, y_encoded, label_encoder, preprocessor


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into training and testing sets with optional stratification."""
    logger.info(f"Splitting dataset: {100 * (1 - test_size):.0f}% train, {100 * test_size:.0f}% test")
    
    # Check if stratification is possible
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_count = class_counts.min()
    
    if min_count >= 2:
        logger.info(f"Using stratified split (all classes have ≥ 2 samples, min={min_count})")
        stratify = y
    else:
        logger.warning(f"Stratified split not possible: {np.sum(class_counts < 2)} classes have < 2 samples. Using random split.")
        stratify = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier."""
    logger.info("Training Decision Tree classifier...")
    try:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        logger.info("Decision Tree training completed")
        return model
    except Exception as e:
        logger.error(f"Decision Tree training failed: {e}")
        raise


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    logger.info("Training Random Forest classifier...")
    try:
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        logger.info("Random Forest training completed")
        return model
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        raise


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a Logistic Regression classifier for multiclass classification."""
    logger.info("Training Logistic Regression classifier...")
    try:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        logger.info("Logistic Regression training completed")
        return model
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {e}")
        raise


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train an XGBoost classifier configured for multiclass classification."""
    logger.info("Training XGBoost classifier...")
    try:
        num_classes_train = len(np.unique(y_train))
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes_train,
            eval_metric="mlogloss",
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        logger.info("XGBoost training completed")
        return model
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        raise


def train_tensorflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int = 50,
    batch_size: int = 32,
) -> tf.keras.Model:
    """Train a TensorFlow Sequential neural network for multiclass classification."""
    logger.info("Building TensorFlow Sequential model...")
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])
        
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        logger.info("TensorFlow model compiled")
        logger.info("Training TensorFlow neural network...")
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )
        
        logger.info("TensorFlow training completed")
        return model
    except Exception as e:
        logger.error(f"TensorFlow training failed: {e}")
        raise


def evaluate_model(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: Optional[LabelEncoder] = None,
) -> Dict[str, Any]:
    """Evaluate model on test set and compute metrics."""
    logger.info(f"Evaluating {model_name}...")
    
    # Get predictions
    if isinstance(model, tf.keras.Model):
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
    else:
        y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Classification report (without target_names to avoid mismatch with test set classes)
    class_report = classification_report(
        y_test, y_pred,
        zero_division=0,
    )
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }
    
    logger.info(
        f"{model_name} metrics: "
        f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
        f"Recall={recall:.4f}, F1={f1:.4f}"
    )
    
    return metrics


def save_best_model(
    model: Any,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save the best model to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, tf.keras.Model):
        model_path = output_dir / "best_model.keras"
        model.save(str(model_path))
        logger.info(f"Saved best model ({model_name}) to {model_path}")
    else:
        model_path = output_dir / "best_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved best model ({model_name}) to {model_path}")
    
    return model_path


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1_score_val = metrics["f1_score"]
        
        print(
            f"{model_name:<25} "
            f"{accuracy:>10.4f}   "
            f"{precision:>10.4f}   "
            f"{recall:>10.4f}   "
            f"{f1_score_val:>10.4f}"
        )
    
    print("=" * 70 + "\n")


def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train multiclass disease prediction models"
    )
    parser.add_argument(
        "--data",
        default=str(default_dataset_path()),
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save models and preprocessors",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    logger.info("="*70)
    logger.info("DISEASE PREDICTION MODEL TRAINING PIPELINE")
    logger.info("="*70)
    
    # 1. Load dataset
    X, y = load_dataset(Path(args.data), args.target)
    
    # 2. Preprocess data
    X_scaled, y_encoded, label_encoder, preprocessor = preprocess_data(X, y, output_dir)
    num_classes = len(label_encoder.classes_)
    
    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(
        X_scaled, y_encoded,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    
    # 4. Train models
    logger.info("="*70)
    logger.info("MODEL TRAINING")
    logger.info("="*70)
    
    results: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Any] = {}
    
    # Decision Tree
    try:
        model = train_decision_tree(X_train, y_train)
        trained_models["Decision Tree"] = model
        results["Decision Tree"] = evaluate_model(
            "Decision Tree", model, X_test, y_test, label_encoder
        )
    except Exception as e:
        logger.error(f"Decision Tree failed, skipping: {e}")
    
    # Random Forest
    try:
        model = train_random_forest(X_train, y_train)
        trained_models["Random Forest"] = model
        results["Random Forest"] = evaluate_model(
            "Random Forest", model, X_test, y_test, label_encoder
        )
    except Exception as e:
        logger.error(f"Random Forest failed, skipping: {e}")
    
    # Logistic Regression
    try:
        model = train_logistic_regression(X_train, y_train)
        trained_models["Logistic Regression"] = model
        results["Logistic Regression"] = evaluate_model(
            "Logistic Regression", model, X_test, y_test, label_encoder
        )
    except Exception as e:
        logger.error(f"Logistic Regression failed, skipping: {e}")
    
    # XGBoost
    try:
        model = train_xgboost(X_train, y_train)
        trained_models["XGBoost"] = model
        results["XGBoost"] = evaluate_model(
            "XGBoost", model, X_test, y_test, label_encoder
        )
    except Exception as e:
        logger.error(f"XGBoost failed, skipping: {e}")
    
    # TensorFlow
    try:
        model = train_tensorflow(X_train, y_train, X_test, y_test, num_classes)
        trained_models["TensorFlow"] = model
        results["TensorFlow"] = evaluate_model(
            "TensorFlow", model, X_test, y_test, label_encoder
        )
    except Exception as e:
        logger.error(f"TensorFlow failed, skipping: {e}")
    
    # 5. Compare models
    logger.info("="*70)
    logger.info("MODEL COMPARISON")
    logger.info("="*70)
    
    if not results:
        logger.error("No models were trained successfully")
        return
    
    print_comparison_table(results)
    
    # 6. Select best model based on accuracy
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_accuracy = results[best_model_name]["accuracy"]
    
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Accuracy: {best_accuracy * 100:.2f}%")
    logger.info("\n" + results[best_model_name]["classification_report"])
    
    # 7. Save best model
    logger.info("="*70)
    logger.info("SAVING BEST MODEL")
    logger.info("="*70)
    
    best_model = trained_models[best_model_name]
    save_best_model(best_model, best_model_name, output_dir)
    if "Random Forest" in trained_models:
        joblib.dump(trained_models["Random Forest"], output_dir / "best_model.pkl")
    
    logger.info("="*70)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)


if __name__ == "__main__":
    main()
