"""
src/optimizer/bess_params_rtc.py — Architecture v10 RTC FINAL (Reverse DC System)
===================================================================================
BESSParamsRTC dataclass for Solar+BESS Reverse DC System with RTC captive contract.

TOPOLOGY: Solar SCBs + BESS in parallel on shared DC Bus → single PCS → AC Bus
  - dc_con:  DC-DC converter capacity (limits solar contribution, replaces S_inv)
  - p_max:   PCS rating (total AC output limit — all flows share this)
  - THRESHOLD = rtc_floor_pct × rtc_mw = 0.80 × 5.0 = 4.0 MW (explicit contract)

KEY CHANGES vs previous version:
  - solar_inverter_mw → dc_con_mw (correct hardware label for Reverse DC)
  - rtc_pshort_threshold_mw → THRESHOLD property (explicit formula)
  - dsm_tol_pct: 0.10 (±10% outer DSM band, used in setpoint outer clamp)
  - discharge_capacity: min(p_max, (soc-e_min)*eta_d/DT)  [s_c removed]
  - C4: soc dynamics driven by x_c (charge) and c_d (discharge) only
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


@dataclass
class BESSParamsRTC:

    # ── BESS Physical ──────────────────────────────────────────────────────
    p_max_mw:                    float   # PCS MW — single PCS, total AC output
    e_max_mwh:                   float
    e_min_mwh:                   float
    eta_charge:                  float
    eta_discharge:               float
    soc_initial_mwh:             float   # SOD = 40 MWh
    soc_terminal_min_mwh:        float   # EOD = 40 MWh (hard equality)

    # ── Financial ─────────────────────────────────────────────────────────
    degradation_cost_rs_mwh:     float
    iex_fee_rs_mwh:              float

    # ── Solar (Reverse DC) ────────────────────────────────────────────────
    solar_capacity_mwp:          float = 25.4
    dc_con_mw:                   float = 16.4   # DC-DC converter capacity

    # ── RTC Contract ──────────────────────────────────────────────────────
    ppa_rate_rs_mwh:             float = 5000.0
    rtc_mw:                      float = 5.0    # contract ceiling = LP upper bound
    rtc_min_mw:                  float = 1.0    # LP lower bound
    rtc_floor_pct:               float = 0.80   # THRESHOLD = rtc_floor_pct × rtc_mw
    rtc_tol_pct:                 float = 0.05   # ±5% inner RTC free band
    dsm_tol_pct:                 float = 0.10   # ±10% outer DSM free band
    rtc_advance_blocks:          int   = 16

    # ── Optimizer ─────────────────────────────────────────────────────────
    max_cycles_per_day:          Optional[float] = 1.0
    soc_terminal_mode:           str   = "hard"
    soc_terminal_value_rs_mwh:   float = 0.0

    # ── Stage 2 ───────────────────────────────────────────────────────────
    rtm_lead_blocks:             int   = 3
    captive_buffer_blocks:       int   = 12
    captive_buffer_tolerance_mw: float = 0.5

    # ── Solar SoC Band ────────────────────────────────────────────────────
    soc_solar_low_pct:           float = 0.15
    soc_solar_high_pct:          float = 0.85
    solar_threshold_mw:          float = 0.5
    solar_buffer_blocks:         int   = 2

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def THRESHOLD(self) -> float:
        """
        Penalty threshold from contract: THRESHOLD = rtc_floor_pct × rtc_mw
        = 0.80 × 5.0 = 4.0 MW (FIXED vs contract ceiling, not daily commitment)
        Used in C_PSHORT and actuals settlement.
        """
        return self.rtc_floor_pct * self.rtc_mw

    @property
    def soc_solar_low(self) -> float:
        return self.soc_solar_low_pct * self.e_max_mwh      # 12.0 MWh

    @property
    def soc_solar_high(self) -> float:
        return self.soc_solar_high_pct * self.e_max_mwh     # 68.0 MWh

    @property
    def avail_cap_mwh(self) -> float:
        """CERC DSM denominator = p_max × DT = 16.4 × 0.25 = 4.1 MWh"""
        return self.p_max_mw * 0.25

    @property
    def usable_energy_mwh(self) -> float:
        """Full BESS usable swing = e_max − e_min = 72.0 MWh"""
        return self.e_max_mwh - self.e_min_mwh

    def discharge_capacity(self, soc: float) -> float:
        """
        Reverse DC discharge capacity (MW).
        Discharge cap = min(p_max, (soc - e_min) × eta_d / DT)
        Note: s_c (solar) is NOT included — solar and BESS are separate DC Bus
        inputs. The LP handles their combined output via C2 (x_d+c_d+s_cd ≤ p_max).
        """
        bess_cap = max(0.0, (soc - self.e_min_mwh) * self.eta_discharge / 0.25)
        return min(self.p_max_mw, bess_cap)

    def rtc_band(self, rtc_committed: float) -> Tuple[float, float]:
        """±5% inner RTC free band — no advance notice needed within this."""
        return (rtc_committed * (1.0 - self.rtc_tol_pct),
                rtc_committed * (1.0 + self.rtc_tol_pct))

    def dsm_band(self, schedule: float) -> Tuple[float, float]:
        """±10% outer DSM free band — penalty-free per CERC DSM 2024."""
        return (schedule * (1.0 - self.dsm_tol_pct),
                schedule * (1.0 + self.dsm_tol_pct))

    @classmethod
    def from_yaml(cls, path: str) -> "BESSParamsRTC":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
