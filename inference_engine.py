"""
Inference Engine Module for Smart Aromatherapy System.

Performs real-time inference on streaming / user physiological data:
1. Validates and preprocesses incoming feature vector
2. Evaluates model probabilities across emotional/stress states
3. Calculates confidence score and dynamic Mood Severity Score (0.0 to 1.0)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import numpy as np
import pandas as pd

from config import MODEL_FILE, PREPROCESSOR_FILE, DEFAULT_ML_CONFIG, MLConfig
from preprocessor import HRVPreprocessor
from model_trainer import ModelTrainer


@dataclass
class InferenceResult:
    """Structured container for single-sample inference results."""
    predicted_state: str
    confidence: float
    probabilities: Dict[str, float]
    severity_score: float  # Normalized 0.0 (baseline calm) to 1.0 (extreme acute stress)
    is_stress: bool
    raw_features: Dict[str, Any] = field(default_factory=dict)
    normalized_features: Optional[np.ndarray] = None


class InferenceEngine:
    """
    Real-time inference engine that transforms physiological data into
    actionable emotion & stress state predictions.
    """

    def __init__(
        self,
        model_file: Union[str, Path] = MODEL_FILE,
        preprocessor_file: Union[str, Path] = PREPROCESSOR_FILE,
        config: MLConfig = DEFAULT_ML_CONFIG,
    ):
        self.model_file = Path(model_file)
        self.preprocessor_file = Path(preprocessor_file)
        self.config = config
        self.preprocessor: Optional[HRVPreprocessor] = None
        self.trainer: Optional[ModelTrainer] = None

        self._initialize()

    def _initialize(self) -> None:
        """Loads or validates model and preprocessor artifacts."""
        if not self.preprocessor_file.exists() or not self.model_file.exists():
            print("[InferenceEngine] Serialized artifacts not found. Initiating auto-training...")
            from model_trainer import train_pipeline
            self.trainer, self.preprocessor = train_pipeline(sample_size=60000, eval_on_test=True, save_artifacts=True)
        else:
            self.preprocessor = HRVPreprocessor.load(self.preprocessor_file)
            self.trainer = ModelTrainer(self.config).load(self.model_file)

    def calculate_severity_score(
        self,
        predicted_state: str,
        probabilities: Dict[str, float],
        raw_features: Dict[str, Any],
    ) -> float:
        """
        Calculates a continuous Mood Severity Score (0.0 to 1.0) combining:
        1. Probabilistic stress mass: P(time pressure) + P(interruption)
        2. Heart Rate (HR) elevation ratio
        3. RMSSD (parasympathetic vagal tone) suppression
        4. LF/HF ratio (sympathovagal balance)
        """
        p_time_pressure = probabilities.get("time pressure", 0.0)
        p_interruption = probabilities.get("interruption", 0.0)
        p_stress_total = p_time_pressure * 1.0 + p_interruption * 0.7

        # Extract physiological telemetry if available
        hr = float(raw_features.get("HR", 72.0))
        rmssd = float(raw_features.get("RMSSD", 35.0))
        lf_hf = float(raw_features.get("LF_HF", 1.5))

        # Physiological perturbation multipliers
        # Normal resting HR: 60-80 bpm; stress > 90 bpm
        hr_factor = np.clip((hr - 65.0) / 45.0, 0.0, 1.0)
        
        # Normal resting RMSSD: 30-60 ms; lower RMSSD (< 20 ms) indicates high sympathetic stress
        rmssd_factor = np.clip((45.0 - rmssd) / 35.0, 0.0, 1.0)

        # Normal LF/HF: 1.0-2.0; acute stress > 3.0
        lf_hf_factor = np.clip((lf_hf - 1.2) / 3.8, 0.0, 1.0)

        physio_stress_index = (0.4 * hr_factor) + (0.35 * rmssd_factor) + (0.25 * lf_hf_factor)

        # Combined severity: 70% model probability + 30% physiological response
        combined_severity = (0.70 * p_stress_total) + (0.30 * physio_stress_index)

        # State bias
        if predicted_state == "no stress":
            combined_severity = min(combined_severity, 0.25)
        elif predicted_state == "interruption":
            combined_severity = np.clip(combined_severity, 0.35, 0.75)
        elif predicted_state == "time pressure":
            combined_severity = np.clip(combined_severity, 0.65, 1.0)

        return float(np.clip(combined_severity, 0.0, 1.0))

    def predict(self, input_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> InferenceResult:
        """
        Runs real-time inference on a single sample of physiological metrics.
        """
        if isinstance(input_data, dict):
            raw_dict = input_data
            df_in = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            raw_dict = input_data.to_dict()
            df_in = pd.DataFrame([raw_dict])
        elif isinstance(input_data, pd.DataFrame):
            df_in = input_data.iloc[[0]]
            raw_dict = df_in.iloc[0].to_dict()
        else:
            raise ValueError(f"Unsupported input format: {type(input_data)}")

        # Transform using preprocessor
        scaled_features = self.preprocessor.transform(df_in)

        # Model inference
        y_pred = self.trainer.model.predict(scaled_features)[0]
        y_prob = self.trainer.model.predict_proba(scaled_features)[0]

        # Map class probabilities
        prob_dict = {
            str(cls_name): float(prob)
            for cls_name, prob in zip(self.trainer.classes, y_prob)
        }

        confidence = float(np.max(y_prob))
        is_stress = y_pred in self.config.stress_states

        severity = self.calculate_severity_score(
            predicted_state=y_pred,
            probabilities=prob_dict,
            raw_features=raw_dict,
        )

        return InferenceResult(
            predicted_state=str(y_pred),
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            severity_score=round(severity, 4),
            is_stress=is_stress,
            raw_features=raw_dict,
            normalized_features=scaled_features[0],
        )
