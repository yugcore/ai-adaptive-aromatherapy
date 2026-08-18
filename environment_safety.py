"""
Environment & Safety Interlocks Module for Smart Aromatherapy System.

Implements:
1. Ambient Temperature & Humidity dynamic dispensing compensation
2. Proximity validation (BLE RSSI signal strength threshold) & Device Authentication
3. Cooldown interval timer (prevents olfactory fatigue and oversaturation)
4. Cumulative daily exposure monitoring (caps total dosage)
5. Piston-aided purge and cleaning cycle scheduler
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np

from config import EnvironmentalSafetyConfig, DEFAULT_SAFETY_CONFIG
from fragrance_controller import DispensingPlan


@dataclass
class EnvironmentalReadings:
    temperature_c: float = 23.0
    humidity_pct: float = 50.0
    air_quality_index: int = 40  # Good (<50)


@dataclass
class SafetyVerificationResult:
    is_safe_to_dispense: bool
    rejection_reason: Optional[str] = None
    adjusted_plan: Optional[DispensingPlan] = None
    triggers_purge_cycle: bool = False
    cooldown_remaining_sec: float = 0.0


class EnvironmentSafetyManager:
    """
    Supervises environmental adaptation and enforces physical safety interlocks.
    """

    def __init__(self, config: EnvironmentalSafetyConfig = DEFAULT_SAFETY_CONFIG):
        self.config = config
        self.last_dispense_timestamp: float = 0.0
        self.cumulative_dispense_ul: float = 0.0
        self.dispense_counter: int = 0
        self.purge_counter: int = 0

    def apply_environmental_compensation(
        self,
        plan: DispensingPlan,
        env: EnvironmentalReadings,
    ) -> DispensingPlan:
        """
        Adjusts spray duration and airflow pressure based on ambient temperature & humidity.
        """
        if not plan.requires_dispense:
            return plan

        # Temperature Delta: Hotter air increases volatility -> reduce spray duration
        temp_delta = env.temperature_c - self.config.baseline_temperature_c
        duration_factor = 1.0 + (temp_delta * self.config.temp_duration_coeff)
        adjusted_duration = max(2.0, plan.spray_duration_sec * duration_factor)

        # Humidity Delta: Higher humidity hinders dispersion -> increase airflow pressure
        humidity_delta = env.humidity_pct - self.config.baseline_humidity_pct
        airflow_factor = 1.0 + (humidity_delta * self.config.humidity_airflow_coeff)
        adjusted_airflow = np.clip(plan.airflow_pressure_kpa * airflow_factor, 25.0, 65.0)

        # Clone and return adjusted plan
        return DispensingPlan(
            action=plan.action,
            target_state=plan.target_state,
            confidence=plan.confidence,
            severity_score=plan.severity_score,
            description=(
                f"{plan.description} [Env Adjusted: Temp {env.temperature_c:.1f}°C (dur x{duration_factor:.2f}), "
                f"Hum {env.humidity_pct:.1f}% (airflow x{airflow_factor:.2f})]"
            ),
            cartridge_doses=plan.cartridge_doses,
            total_dose_ul=plan.total_dose_ul,
            spray_duration_sec=round(float(adjusted_duration), 2),
            airflow_pressure_kpa=round(float(adjusted_airflow), 1),
            nozzle_duty_cycle_pct=plan.nozzle_duty_cycle_pct,
            is_multi_blend=plan.is_multi_blend,
            requires_dispense=plan.requires_dispense,
        )

    def check_proximity_and_auth(
        self,
        device_id: str,
        rssi_dbm: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates whether the wearable smart band is within range and whitelisted.
        """
        if self.config.require_authentication:
            if device_id not in self.config.authorized_device_ids:
                return False, f"Device ID '{device_id}' unauthorized. Access denied."

        if rssi_dbm < self.config.proximity_rssi_threshold:
            return (
                False,
                f"User out of proximity range (RSSI: {rssi_dbm:.1f} dBm < Threshold {self.config.proximity_rssi_threshold} dBm).",
            )

        return True, None

    def verify_and_adapt(
        self,
        plan: DispensingPlan,
        env: Optional[EnvironmentalReadings] = None,
        device_id: str = "SMARTBAND_AGY_001",
        rssi_dbm: float = -62.0,
        proximity_rssi_dbm: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> SafetyVerificationResult:
        """
        Performs full safety interlock checks and applies environmental compensation.
        """
        now = current_time or time.time()
        env_reading = env or EnvironmentalReadings()
        effective_rssi = proximity_rssi_dbm if proximity_rssi_dbm is not None else rssi_dbm

        # Check 1: Does the plan require dispensing?
        if not plan.requires_dispense:
            return SafetyVerificationResult(
                is_safe_to_dispense=False,
                rejection_reason="Plan does not require active fragrance dispensing.",
                adjusted_plan=plan,
            )

        # Check 2: Proximity & Authentication
        is_auth, auth_err = self.check_proximity_and_auth(device_id, effective_rssi)
        if not is_auth:
            return SafetyVerificationResult(
                is_safe_to_dispense=False,
                rejection_reason=auth_err,
                adjusted_plan=plan,
            )

        # Check 3: Cooldown interval
        time_since_last = now - self.last_dispense_timestamp
        if self.last_dispense_timestamp > 0 and time_since_last < self.config.min_cooldown_seconds:
            remaining = self.config.min_cooldown_seconds - time_since_last
            return SafetyVerificationResult(
                is_safe_to_dispense=False,
                rejection_reason=f"Cooldown active ({remaining:.1f}s remaining). Preventing olfactory oversaturation.",
                adjusted_plan=plan,
                cooldown_remaining_sec=remaining,
            )

        # Check 4: Cumulative Daily Dosage
        if (self.cumulative_dispense_ul + plan.total_dose_ul) > self.config.max_daily_dispense_ul:
            return SafetyVerificationResult(
                is_safe_to_dispense=False,
                rejection_reason=f"Daily exposure limit exceeded ({self.cumulative_dispense_ul:.1f} / {self.config.max_daily_dispense_ul} µL).",
                adjusted_plan=plan,
            )

        # Apply Environmental Compensation
        adapted_plan = self.apply_environmental_compensation(plan, env_reading)

        # Check Purge Cycle requirement
        triggers_purge = False
        if (self.dispense_counter + 1) % self.config.purge_cycle_frequency == 0:
            triggers_purge = True

        return SafetyVerificationResult(
            is_safe_to_dispense=True,
            adjusted_plan=adapted_plan,
            triggers_purge_cycle=triggers_purge,
            cooldown_remaining_sec=0.0,
        )

    def record_dispense_event(self, dose_ul: float, timestamp: Optional[float] = None) -> None:
        """Updates internal state after a verified dispensing execution."""
        now = timestamp or time.time()
        self.last_dispense_timestamp = now
        self.cumulative_dispense_ul += dose_ul
        self.dispense_counter += 1

    def record_purge_event(self) -> None:
        """Updates purge counter."""
        self.purge_counter += 1

    def reset_daily_stats(self) -> None:
        """Resets daily cumulative dose."""
        self.cumulative_dispense_ul = 0.0
        print("[EnvironmentSafety] Daily exposure stats reset.")
