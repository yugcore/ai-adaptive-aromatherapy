"""
Fragrance Controller Module for Smart Aromatherapy System.

Translates inferred emotional state and Mood Severity Score into:
1. Rotary cartridge selection (1 to 6 magazine slots)
2. Single & Multi-fragrance blending compositions
3. Dynamic dispensing parameters (dosage volume, spray duration, pneumatic airflow, duty cycle)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from config import (
    FRAGRANCE_MAGAZINE,
    STATE_TO_FRAGRANCE_MAP,
    FragranceCartridge,
    DEFAULT_ML_CONFIG,
)
from inference_engine import InferenceResult


@dataclass
class CartridgeDose:
    slot_id: int
    fragrance_name: str
    botanical_name: str
    dose_ul: float  # microliters
    blend_percentage: float  # 0 to 100%


@dataclass
class DispensingPlan:
    action: str  # "neutralize", "maintain", "none", "focus"
    target_state: str
    confidence: float
    severity_score: float
    description: str
    cartridge_doses: List[CartridgeDose]
    total_dose_ul: float
    spray_duration_sec: float
    airflow_pressure_kpa: float
    nozzle_duty_cycle_pct: float
    is_multi_blend: bool
    requires_dispense: bool


class FragranceController:
    """
    Decides and generates fine-grained pneumatic dispensing recipes based on
    emotion classification and severity scores.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_ML_CONFIG.confidence_threshold):
        self.confidence_threshold = confidence_threshold
        self.magazine = FRAGRANCE_MAGAZINE
        self.state_map = STATE_TO_FRAGRANCE_MAP

    def generate_dispensing_plan(self, inference: InferenceResult) -> DispensingPlan:
        """
        Calculates the complete dispensing recipe based on inference result.
        """
        state = inference.predicted_state
        confidence = inference.confidence
        severity = inference.severity_score

        # Rule 1: Low confidence check
        if confidence < self.confidence_threshold:
            return DispensingPlan(
                action="none",
                target_state=state,
                confidence=confidence,
                severity_score=severity,
                description=f"Inference confidence ({confidence:.2f}) below safety threshold ({self.confidence_threshold:.2f}). Dispensing suppressed.",
                cartridge_doses=[],
                total_dose_ul=0.0,
                spray_duration_sec=0.0,
                airflow_pressure_kpa=0.0,
                nozzle_duty_cycle_pct=0.0,
                is_multi_blend=False,
                requires_dispense=False,
            )

        # Retrieve mapped profile or default fallback
        profile = self.state_map.get(
            state,
            {
                "action": "maintain",
                "primary_cartridge": 6,
                "secondary_cartridge": None,
                "blend_ratio": 1.0,
                "intensity_multiplier": 0.5,
                "description": "Default maintenance profile.",
            },
        )

        action = profile["action"]
        primary_slot = profile["primary_cartridge"]
        secondary_slot = profile.get("secondary_cartridge")
        blend_ratio = profile.get("blend_ratio", 1.0)
        base_intensity = profile.get("intensity_multiplier", 1.0)

        # Primary cartridge specification
        primary_cart = self.magazine[primary_slot]

        # Dynamic intensity modulation factor (proportional to severity score)
        # Severity ranges from 0.0 (calm) to 1.0 (peak acute stress)
        dynamic_mod = 0.6 + (0.8 * severity) * base_intensity

        # Compute modulated delivery parameters
        base_dose = primary_cart.base_dosage_ul * dynamic_mod
        base_duration = primary_cart.base_duration_sec * np.clip(dynamic_mod, 0.7, 1.5)
        base_pressure = primary_cart.base_airflow_kpa * np.clip(0.85 + 0.3 * severity, 0.8, 1.3)
        duty_cycle = np.clip(50.0 + (severity * 45.0), 40.0, 95.0)

        # Cartridge dose breakdown
        cartridge_doses = []
        is_multi_blend = False

        if secondary_slot is not None and blend_ratio < 1.0:
            is_multi_blend = True
            sec_cart = self.magazine[secondary_slot]
            primary_vol = base_dose * blend_ratio
            sec_vol = base_dose * (1.0 - blend_ratio)

            cartridge_doses.append(
                CartridgeDose(
                    slot_id=primary_slot,
                    fragrance_name=primary_cart.name,
                    botanical_name=primary_cart.botanical_name,
                    dose_ul=round(primary_vol, 1),
                    blend_percentage=round(blend_ratio * 100, 1),
                )
            )
            cartridge_doses.append(
                CartridgeDose(
                    slot_id=secondary_slot,
                    fragrance_name=sec_cart.name,
                    botanical_name=sec_cart.botanical_name,
                    dose_ul=round(sec_vol, 1),
                    blend_percentage=round((1.0 - blend_ratio) * 100, 1),
                )
            )
        else:
            cartridge_doses.append(
                CartridgeDose(
                    slot_id=primary_slot,
                    fragrance_name=primary_cart.name,
                    botanical_name=primary_cart.botanical_name,
                    dose_ul=round(base_dose, 1),
                    blend_percentage=100.0,
                )
            )

        total_dose = sum(c.dose_ul for c in cartridge_doses)

        return DispensingPlan(
            action=action,
            target_state=state,
            confidence=confidence,
            severity_score=severity,
            description=profile["description"],
            cartridge_doses=cartridge_doses,
            total_dose_ul=round(total_dose, 1),
            spray_duration_sec=round(float(base_duration), 2),
            airflow_pressure_kpa=round(float(base_pressure), 1),
            nozzle_duty_cycle_pct=round(float(duty_cycle), 1),
            is_multi_blend=is_multi_blend,
            requires_dispense=True,
        )
