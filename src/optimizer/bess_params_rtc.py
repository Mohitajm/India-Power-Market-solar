"""
src/optimizer/bess_params_rtc.py — Architecture v10 RTC FINAL (Reverse DC System)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml

@dataclass
class BESSParamsRTC:
    # BESS Physical
    p_max_mw:                    float
    e_max_mwh:                   float
    e_min_mwh:                   float
    eta_charge:                  float
    eta_discharge:               float
    soc_initial_mwh:             float
    soc_terminal_min_mwh:        float
    # Financial
    ppa_rate_rs_mwh:             float
    iex_fee_rs_mwh:              float
    degradation_cost_rs_mwh:     float
    # Solar (Reverse DC)
    solar_capacity_mwp:          float = 25.4
    dc_con_mw:                   float = 16.4
    # RTC Contract
    rtc_mw:                      float = 5.0
    rtc_min_mw:                  float = 1.0
    rtc_floor_pct:               float = 0.80
    rtc_tol_pct:                 float = 0.05
    dsm_tol_pct:                 float = 0.10
    rtc_advance_blocks:          int   = 16
    # Optimizer
    max_cycles_per_day:          Optional[float] = 1.0
    soc_terminal_mode:           str   = "hard"
    soc_terminal_value_rs_mwh:   float = 0.0
    # Stage 2
    rtm_lead_blocks:             int   = 3
    captive_buffer_blocks:       int   = 12
    captive_buffer_tolerance_mw: float = 0.5
    # Solar SoC Band
    soc_solar_low_pct:           float = 0.15
    soc_solar_high_pct:          float = 0.85
    solar_threshold_mw:          float = 0.5
    solar_buffer_blocks:         int   = 2

    @property
    def THRESHOLD(self) -> float:
        """THRESHOLD = rtc_floor_pct × rtc_mw = 0.80 × 5.0 = 4.0 MW (FIXED vs ceiling)"""
        return self.rtc_floor_pct * self.rtc_mw

    @property
    def soc_solar_low(self) -> float:
        return self.soc_solar_low_pct * self.e_max_mwh

    @property
    def soc_solar_high(self) -> float:
        return self.soc_solar_high_pct * self.e_max_mwh

    @property
    def usable_energy_mwh(self) -> float:
        return self.e_max_mwh - self.e_min_mwh

    @property
    def avail_cap_mwh(self) -> float:
        return self.p_max_mw * 0.25

    def rtc_band(self, rtc_committed: float) -> Tuple[float, float]:
        return (rtc_committed*(1.0-self.rtc_tol_pct), rtc_committed*(1.0+self.rtc_tol_pct))

    def dsm_band(self, schedule: float) -> Tuple[float, float]:
        return (schedule*(1.0-self.dsm_tol_pct), schedule*(1.0+self.dsm_tol_pct))

    @classmethod
    def from_yaml(cls, path: str) -> "BESSParamsRTC":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
