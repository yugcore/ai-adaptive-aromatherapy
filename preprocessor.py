"""
Preprocessor Module for Smart Aromatherapy System.

Handles:
- Data type coercion to numeric
- Empty / all-NaN column detection & removal
- Infinite (+/- inf) value handling
- Missing value imputation (SimpleImputer)
- Feature standardization (StandardScaler)
- Serialization / Deserialization of preprocessor pipeline using joblib
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import PREPROCESSOR_FILE


class HRVPreprocessor:
    """
    Stateful preprocessor for HRV physiological features.
    Maintains fitted imputer, scaler, and expected feature column definitions.
    """

    def __init__(self, imputer_strategy: str = "mean"):
        self.imputer_strategy = imputer_strategy
        self.imputer = SimpleImputer(strategy=imputer_strategy)
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.dropped_empty_columns: List[str] = []
        self.is_fitted: bool = False

    def _clean_dataframe(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Coerces to numeric, strips infinite values, and drops or aligns columns."""
        df_clean = df.copy()

        # Coerce all columns to numeric, replacing unparseable items with NaN
        df_clean = df_clean.apply(pd.to_numeric, errors="coerce")

        if is_training:
            # Find and record completely empty columns
            all_nan_cols = df_clean.columns[df_clean.isnull().all()].tolist()
            if all_nan_cols:
                print(f"[Preprocessor] Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
                self.dropped_empty_columns = all_nan_cols
                df_clean = df_clean.drop(columns=all_nan_cols)
            self.feature_columns = list(df_clean.columns)
        else:
            # Drop the columns that were dropped during training
            if self.dropped_empty_columns:
                existing_drop_cols = [c for c in self.dropped_empty_columns if c in df_clean.columns]
                if existing_drop_cols:
                    df_clean = df_clean.drop(columns=existing_drop_cols)

            # Ensure all expected feature columns exist (fill with NaN if missing)
            for col in self.feature_columns:
                if col not in df_clean.columns:
                    df_clean[col] = np.nan

            # Reorder columns to strictly match training order
            df_clean = df_clean[self.feature_columns]

        # Replace infinite values with NaN
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_clean

    def fit(self, X: pd.DataFrame) -> "HRVPreprocessor":
        """Fits the imputer and scaler on the training features."""
        print(f"[Preprocessor] Fitting preprocessor on {X.shape[0]:,} samples and {X.shape[1]} raw features...")
        X_clean = self._clean_dataframe(X, is_training=True)

        X_imputed = self.imputer.fit_transform(X_clean)
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        print(f"[Preprocessor] Successfully fitted. Final feature count: {len(self.feature_columns)}")
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, Dict[str, Any], np.ndarray]) -> np.ndarray:
        """Transforms new input data into scaled numeric arrays using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted or loaded yet! Call fit() or load() first.")

        # Convert dict or series to DataFrame
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            df = pd.DataFrame([X.to_dict()])
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_columns if len(self.feature_columns) == X.shape[1] else None)
        else:
            raise ValueError(f"Unsupported input type for transformation: {type(X)}")

        df_clean = self._clean_dataframe(df, is_training=False)
        X_imputed = self.imputer.transform(df_clean)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits and transforms training data in a single step."""
        self.fit(X)
        X_clean = self._clean_dataframe(X, is_training=False)
        X_imputed = self.imputer.transform(X_clean)
        return self.scaler.transform(X_imputed)

    def save(self, filepath: Union[str, Path] = PREPROCESSOR_FILE) -> None:
        """Persists the preprocessor object to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Preprocessor] Saved fitted preprocessor to: {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Union[str, Path] = PREPROCESSOR_FILE) -> "HRVPreprocessor":
        """Loads a preprocessor object from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Fitted preprocessor not found at {filepath.resolve()}")
        preprocessor = joblib.load(filepath)
        print(f"[Preprocessor] Loaded preprocessor from: {filepath.resolve()}")
        return preprocessor
