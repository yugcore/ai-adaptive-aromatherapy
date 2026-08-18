"""
Data Loader Module for Smart Aromatherapy System.

Provides robust, modular data ingestion for:
1. HRV Stress Dataset (train.csv and test.csv)
2. Physiological Stress Detection Dataset (stress_detection.csv)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np

from config import DATASET_PATHS, MLConfig, DEFAULT_ML_CONFIG


class DatasetLoader:
    """Handles loading, path resolution, schema validation, and subset extraction for datasets."""

    def __init__(self, config: MLConfig = DEFAULT_ML_CONFIG):
        self.config = config

    def check_file_exists(self, file_key: str) -> Path:
        """Verifies if the configured dataset file exists on disk."""
        path = DATASET_PATHS.get(file_key)
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"Dataset '{file_key}' not found at expected path: {path}.\n"
                f"Please ensure dataset files are placed inside the respective folders."
            )
        return Path(path)

    def load_hrv_train(
        self,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads the HRV training dataset.

        Args:
            sample_size: If specified, samples a subset of rows for faster prototyping.
            random_state: Seed for reproducible random sampling.

        Returns:
            Tuple of (X_train: pd.DataFrame, y_train: pd.Series)
        """
        train_path = self.check_file_exists("hrv_train")
        print(f"[DataLoader] Loading HRV training data from: {train_path.resolve()}")

        df = pd.read_csv(train_path)
        print(f"[DataLoader] Loaded {len(df):,} total rows and {len(df.columns)} columns.")

        if sample_size and sample_size < len(df):
            print(f"[DataLoader] Subsampling {sample_size:,} rows for rapid execution...")
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

        if self.config.target_column not in df.columns:
            raise KeyError(f"Target column '{self.config.target_column}' not found in dataset!")

        y_train = df[self.config.target_column]
        
        # Drop unwanted columns
        drop_cols = [col for col in self.config.drop_columns if col in df.columns]
        X_train = df.drop(columns=drop_cols)

        print(f"[DataLoader] Features shape: {X_train.shape}, Target distribution:")
        for label, count in y_train.value_counts().items():
            print(f"   - {label}: {count:,} ({count / len(y_train) * 100:.1f}%)")

        return X_train, y_train

    def load_hrv_test(
        self,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Loads the HRV test dataset.

        Returns:
            Tuple of (X_test: pd.DataFrame, y_test: Optional[pd.Series])
        """
        test_path = self.check_file_exists("hrv_test")
        print(f"[DataLoader] Loading HRV test data from: {test_path.resolve()}")

        df = pd.read_csv(test_path)
        print(f"[DataLoader] Loaded test dataset with {len(df):,} rows.")

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

        y_test = None
        if self.config.target_column in df.columns:
            y_test = df[self.config.target_column]

        drop_cols = [col for col in self.config.drop_columns if col in df.columns]
        X_test = df.drop(columns=drop_cols)

        return X_test, y_test

    def load_physiological_stress_data(self) -> pd.DataFrame:
        """
        Loads the physiological multimodal stress detection dataset (PSS, GSR/skin conductance,
        accelerometer, sleep, and behavioral metrics).
        """
        physio_path = self.check_file_exists("physiological_stress")
        print(f"[DataLoader] Loading physiological stress data from: {physio_path.resolve()}")

        df = pd.read_csv(physio_path)
        print(f"[DataLoader] Loaded {len(df):,} records with columns: {list(df.columns)}")
        return df

    def get_feature_names(self) -> List[str]:
        """Returns the expected HRV feature column names by inspecting the dataset header."""
        train_path = self.check_file_exists("hrv_train")
        df_sample = pd.read_csv(train_path, nrows=2)
        drop_cols = [c for c in self.config.drop_columns if c in df_sample.columns]
        return [c for c in df_sample.columns if c not in drop_cols]
