"""
Central Configuration Module for Smart Aromatherapy System.

Defines:
- Dynamic paths to all datasets (HRV stress dataset, physiological stress detection dataset, wearable HRV emotion dataset)
- Model hyperparameters and serialized model paths
- Fragrance cartridge specifications and emotion-to-fragrance mappings
- Environmental compensation coefficients (temperature, humidity)
- Safety thresholds (proximity RSSI, cooldown times, cumulative daily dosage limits, purge intervals)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any

# ==============================================================================
# BASE DIRECTORIES & PATH RESOLUTION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Dataset Directories
DATASETS_DIR = {
    "hrv_stress": BASE_DIR / "HRV stress dataset",
    "physiological_stress": BASE_DIR / "physiological stress detection dataset",
    "wearable_hrv": BASE_DIR / "wearable HRV emotion dataset",
}

# Individual File Paths
DATASET_PATHS = {
    "hrv_train": DATASETS_DIR["hrv_stress"] / "train.csv",
    "hrv_test": DATASETS_DIR["hrv_stress"] / "test.csv",
    "physiological_stress": DATASETS_DIR["physiological_stress"] / "stress_detection.csv",
}

# Model and Artifact Paths
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "hrv_rf_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "hrv_preprocessor.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"


# ==============================================================================
# MACHINE LEARNING CONFIGURATION
# ==============================================================================
@dataclass
class MLConfig:
    """Hyperparameters and configuration for ML pipeline."""
    target_column: str = "condition"
    drop_columns: List[str] = field(default_factory=lambda: ["datasetId", "condition"])
    imputer_strategy: str = "mean"
    n_estimators: int = 100
    max_depth: int = 25
    min_samples_split: int = 5
    random_state: int = 42
    n_jobs: int = -1
    stress_states: List[str] = field(default_factory=lambda: ["time pressure", "interruption"])
    confidence_threshold: float = 0.60


# ==============================================================================
# FRAGRANCE CATALOG & CARTRIDGE CONFIGURATION
# ==============================================================================
@dataclass
class FragranceCartridge:
    slot_id: int
    name: str
    botanical_name: str
    primary_benefit: str
    target_emotions: List[str]
    base_dosage_ul: float  # microliters
    base_duration_sec: float  # seconds
    base_airflow_kpa: float  # kPa


# Rotary Magazine Slots (6-Cartridge Chamber as specified in patent)
FRAGRANCE_MAGAZINE: Dict[int, FragranceCartridge] = {
    1: FragranceCartridge(
        slot_id=1,
        name="Lavender (French)",
        botanical_name="Lavandula angustifolia",
        primary_benefit="Sedative, anxiolytic, parasympathetic activator",
        target_emotions=["time pressure", "high stress", "anxiety"],
        base_dosage_ul=120.0,
        base_duration_sec=8.0,
        base_airflow_kpa=45.0,
    ),
    2: FragranceCartridge(
        slot_id=2,
        name="Roman Chamomile",
        botanical_name="Chamaemelum nobile",
        primary_benefit="Mild calming, soothing nervous tension",
        target_emotions=["interruption", "mild stress", "restlessness"],
        base_dosage_ul=90.0,
        base_duration_sec=6.0,
        base_airflow_kpa=38.0,
    ),
    3: FragranceCartridge(
        slot_id=3,
        name="Sweet Orange & Bergamot",
        botanical_name="Citrus sinensis / Citrus bergamia",
        primary_benefit="Mood uplifting, cortisol reduction, mild alertness",
        target_emotions=["fatigue", "low mood", "sluggishness"],
        base_dosage_ul=100.0,
        base_duration_sec=7.0,
        base_airflow_kpa=40.0,
    ),
    4: FragranceCartridge(
        slot_id=4,
        name="Eucalyptus Globulus",
        botanical_name="Eucalyptus globulus",
        primary_benefit="Cognitive clarity, respiratory expansion, alertness",
        target_emotions=["brain fog", "cognitive overload"],
        base_dosage_ul=80.0,
        base_duration_sec=5.0,
        base_airflow_kpa=42.0,
    ),
    5: FragranceCartridge(
        slot_id=5,
        name="Peppermint",
        botanical_name="Mentha piperita",
        primary_benefit="Instant stimulation, headache relief, focus booster",
        target_emotions=["drowsiness", "mid-day slump"],
        base_dosage_ul=75.0,
        base_duration_sec=4.5,
        base_airflow_kpa=40.0,
    ),
    6: FragranceCartridge(
        slot_id=6,
        name="Sandalwood & Frankincense",
        botanical_name="Santalum album / Boswellia carterii",
        primary_benefit="Deep meditative grounding, homeostasis maintenance",
        target_emotions=["no stress", "baseline", "meditation"],
        base_dosage_ul=60.0,
        base_duration_sec=4.0,
        base_airflow_kpa=35.0,
    ),
}

# State to Fragrance Mapping
STATE_TO_FRAGRANCE_MAP: Dict[str, Dict[str, Any]] = {
    "time pressure": {
        "action": "neutralize",
        "primary_cartridge": 1,  # Lavender
        "secondary_cartridge": 2,  # Chamomile blend
        "blend_ratio": 0.80,  # 80% primary, 20% secondary
        "intensity_multiplier": 1.25,
        "description": "High acute stress detected. Dispensing deep-calm anxiolytic blend.",
    },
    "interruption": {
        "action": "neutralize",
        "primary_cartridge": 2,  # Chamomile
        "secondary_cartridge": 3,  # Bergamot
        "blend_ratio": 0.70,
        "intensity_multiplier": 1.0,
        "description": "Cognitive interruption/mild agitation detected. Dispensing soothing stabilizing blend.",
    },
    "no stress": {
        "action": "maintain",
        "primary_cartridge": 6,  # Sandalwood
        "secondary_cartridge": None,
        "blend_ratio": 1.0,
        "intensity_multiplier": 0.5,
        "description": "Homeostatic balance maintained. Ambient grounding fragrance emission.",
    },
}


# ==============================================================================
# ENVIRONMENTAL ADAPTATION & SAFETY INTERLOCKS CONFIG
# ==============================================================================
@dataclass
class EnvironmentalSafetyConfig:
    """Environmental sensor compensation thresholds and safety interlock parameters."""
    # Baseline environment conditions
    baseline_temperature_c: float = 23.0  # Celsius
    baseline_humidity_pct: float = 50.0  # % Relative Humidity
    
    # Sensitivity coefficients
    temp_duration_coeff: float = -0.02  # Higher temp = shorter spray (faster volatile evaporation)
    humidity_airflow_coeff: float = 0.015  # Higher humidity = higher airflow pressure to promote dispersion
    
    # Proximity parameters (BLE RSSI)
    proximity_rssi_threshold: float = -75.0  # dBm (closer than approx 3-4 meters)
    require_authentication: bool = True
    authorized_device_ids: List[str] = field(default_factory=lambda: ["SMARTBAND_AGY_001", "TEST_BAND_DEBUG"])
    
    # Cooldown & Exposure protection
    min_cooldown_seconds: float = 60.0  # 1 minute between sprays
    max_cooldown_seconds: float = 300.0  # 5 minutes after intense session
    max_daily_dispense_ul: float = 5000.0  # 5 mL max per day
    
    # Purge cycle
    purge_cycle_frequency: int = 10  # Auto-purge residual every 10 dispenses
    purge_duration_sec: float = 3.0
    purge_airflow_kpa: float = 60.0


# Single global instances
DEFAULT_ML_CONFIG = MLConfig()
DEFAULT_SAFETY_CONFIG = EnvironmentalSafetyConfig()
