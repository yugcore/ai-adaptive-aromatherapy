"""
Model Trainer Module for Smart Aromatherapy System.

Trains, tunes, evaluates, and persists Machine Learning models for
physiological stress and emotional state classification from HRV features.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    MLConfig,
    DEFAULT_ML_CONFIG,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    METADATA_FILE,
)
from data_loader import DatasetLoader
from preprocessor import HRVPreprocessor


class ModelTrainer:
    """
    Manages model initialization, training, evaluation, and disk persistence.
    """

    def __init__(self, config: MLConfig = DEFAULT_ML_CONFIG):
        self.config = config
        self.model: Optional[RandomForestClassifier] = None
        self.classes: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None
        self.training_summary: Dict[str, Any] = {}

    def build_model(self) -> RandomForestClassifier:
        """Instantiates the Random Forest Classifier according to config."""
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

    def train(
        self,
        X_train_scaled: np.ndarray,
        y_train: pd.Series,
        feature_names: Optional[list] = None,
    ) -> RandomForestClassifier:
        """
        Trains the Random Forest model on preprocessed feature array.
        """
        print(f"[ModelTrainer] Initializing RandomForest (n_estimators={self.config.n_estimators})...")
        self.model = self.build_model()
        self.feature_names = feature_names

        start_time = time.time()
        print(f"[ModelTrainer] Fitting model on {X_train_scaled.shape[0]:,} samples...")
        self.model.fit(X_train_scaled, y_train)
        elapsed_sec = time.time() - start_time

        self.classes = self.model.classes_
        print(f"[ModelTrainer] Training complete in {elapsed_sec:.2f}s! Target classes: {list(self.classes)}")

        self.training_summary["train_samples"] = int(X_train_scaled.shape[0])
        self.training_summary["train_features"] = int(X_train_scaled.shape[1])
        self.training_summary["training_time_sec"] = round(elapsed_sec, 2)
        self.training_summary["classes"] = list(self.classes)

        return self.model

    def evaluate(
        self,
        X_test_scaled: np.ndarray,
        y_test: pd.Series,
        dataset_name: str = "Test Set",
    ) -> Dict[str, Any]:
        """
        Evaluates the trained model against ground truth labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet!")

        print(f"\n[ModelTrainer] --- Evaluating Model on {dataset_name} ({len(y_test):,} samples) ---")
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"[ModelTrainer] Accuracy:           {acc * 100:.2f}%")
        print(f"[ModelTrainer] Weighted F1-Score:   {f1_weighted:.4f}")
        print(f"[ModelTrainer] Macro F1-Score:      {f1_macro:.4f}")
        print(f"[ModelTrainer] Precision:          {precision:.4f}")
        print(f"[ModelTrainer] Recall:             {recall:.4f}")
        print("\n[ModelTrainer] Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        print("[ModelTrainer] Confusion Matrix:")
        cm_df = pd.DataFrame(cm, index=self.classes, columns=self.classes)
        print(cm_df)

        metrics = {
            "dataset": dataset_name,
            "sample_count": len(y_test),
            "accuracy": round(float(acc), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "confusion_matrix": cm.tolist(),
        }
        self.training_summary["evaluation"] = metrics
        return metrics

    def save(
        self,
        model_path: Union[str, Path] = MODEL_FILE,
        metadata_path: Union[str, Path] = METADATA_FILE,
    ) -> None:
        """Saves trained model and metadata."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model!")

        model_path = Path(model_path)
        metadata_path = Path(metadata_path)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"[ModelTrainer] Saved model to: {model_path.resolve()}")

        # Save metadata
        meta = {
            "classes": [str(c) for c in self.classes] if self.classes is not None else [],
            "feature_names": self.feature_names or [],
            "config": {
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "min_samples_split": self.config.min_samples_split,
                "random_state": self.config.random_state,
            },
            "summary": self.training_summary,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[ModelTrainer] Saved metadata to: {metadata_path.resolve()}")

    def load(
        self,
        model_path: Union[str, Path] = MODEL_FILE,
        metadata_path: Union[str, Path] = METADATA_FILE,
    ) -> "ModelTrainer":
        """Loads serialized model and metadata."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path.resolve()}")

        self.model = joblib.load(model_path)
        self.classes = getattr(self.model, "classes_", np.array(["no stress", "interruption", "time pressure"]))

        if Path(metadata_path).exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.training_summary = meta.get("summary", {})

        print(f"[ModelTrainer] Loaded trained model with classes: {list(self.classes)}")
        return self


def train_pipeline(
    sample_size: Optional[int] = 100000,
    eval_on_test: bool = True,
    save_artifacts: bool = True,
) -> Tuple[ModelTrainer, HRVPreprocessor]:
    """
    End-to-end training pipeline function.
    Loads data -> Preprocesses -> Trains Random Forest -> Evaluates -> Saves artifacts.
    """
    print("=" * 70)
    print("STARTING SMART AROMATHERAPY ML MODEL TRAINING PIPELINE")
    print("=" * 70)

    loader = DatasetLoader()
    X_train, y_train = loader.load_hrv_train(sample_size=sample_size)

    preprocessor = HRVPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)

    trainer = ModelTrainer()
    trainer.train(X_train_scaled, y_train, feature_names=preprocessor.feature_columns)

    if eval_on_test:
        X_test, y_test = loader.load_hrv_test(sample_size=20000 if sample_size else None)
        if y_test is not None:
            X_test_scaled = preprocessor.transform(X_test)
            trainer.evaluate(X_test_scaled, y_test, dataset_name="Test.csv")

    if save_artifacts:
        preprocessor.save(PREPROCESSOR_FILE)
        trainer.save(MODEL_FILE, METADATA_FILE)

    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return trainer, preprocessor
