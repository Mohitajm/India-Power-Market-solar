"""
src/optimizer/bess_params_rtc.py — Architecture v10 RTC FINAL
==============================================================
BESSParamsRTC dataclass for Solar+BESS RTC captive contract.

Hardware:  25.4 MWp DC / 16.4 MW AC inverter / 80 MWh BESS / 16.4 MW PCS
Contract:  5 MW RTC ceiling | PPA Rs 5,000/MWh
SOD=EOD:   40.0 MWh hard equality (C5 in all three LP stages)
Topology:  DC-coupled — solar and BESS share DC Bus; single PCS → AC Bus

Key parameters vs v9 bess_params.py:
  p_max_mw:               2.5   → 16.4  MW
  e_max_mwh:              4.75  → 80.0  MWh
  e_min_mwh:              0.50  → 8.0   MWh
  soc_initial_mwh:        2.50  → 40.0  MWh (SOD = EOD = 40 MWh)
  ppa_rate_rs_mwh:        3500  → 5000  Rs/MWh
  solar_inverter_mw:      25.0  → 16.4  MW
  solar_capacity_mwp:     35.0  → 25.4  MWp
  rtc_mw/min_mw:          new   → 5.0 / 1.0 MW
  rtc_floor_pct:          new   → 0.80  (fixed 4 MW C_PSHORT threshold)
  rtc_tol_pct:            new   → 0.05  (±5% free RT band with setpoint clamp)
  rtc_advance_blocks:     new   → 16    (4-hour notice for >5% revision)
  max_cycles_per_day:     null  → 1     (C7: USABLE = 72 MWh / day)
  soc_terminal_mode:      hard  → hard  (== 40 MWh)
  soc_terminal_value:     0     → 0     (removed; hard equality replaces it)
  soc_solar_low_pct:      0.20  → 0.15  (12 MWh during solar)
  soc_solar_high_pct:     0.80  → 0.85  (68 MWh during solar)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


@dataclass
class BESSParamsRTC:

    # ── BESS Physical ──────────────────────────────────────────────────────
    p_max_mw:                    float   # PCS MW (charge or discharge)
    e_max_mwh:                   float   # BESS ceiling MWh
    e_min_mwh:                   float   # BESS floor MWh (10% DoD)
    eta_charge:                  float
    eta_discharge:               float
    soc_initial_mwh:             float   # SOD (= 40 MWh; EOD chained)
    soc_terminal_min_mwh:        float   # EOD hard equality target (40 MWh)

    # ── Financial ─────────────────────────────────────────────────────────
    degradation_cost_rs_mwh:     float   # Post-hoc only — NOT in LP
    iex_fee_rs_mwh:              float

    # ── Solar ─────────────────────────────────────────────────────────────
    solar_capacity_mwp:          float = 25.4
    solar_inverter_mw:           float = 16.4

    # ── RTC Contract ──────────────────────────────────────────────────────
    ppa_rate_rs_mwh:             float = 5000.0
    rtc_mw:                      float = 5.0    # contract ceiling = LP upper bound
    rtc_min_mw:                  float = 1.0    # LP lower bound (very low SoC days)
    rtc_floor_pct:               float = 0.80   # fixed penalty threshold = 0.80×5=4 MW
    rtc_tol_pct:                 float = 0.05   # ±5% free RT band; setpoint clamp
    rtc_advance_blocks:          int   = 16     # 4-hour advance notice for >5% revision

    # ── Optimizer ─────────────────────────────────────────────────────────
    max_cycles_per_day:          Optional[float] = 1.0   # C7: 1 cycle = 72 MWh
    soc_terminal_mode:           str   = "hard"          # == 40 MWh equality
    soc_terminal_value_rs_mwh:   float = 0.0             # 0 — hard equality replaces it

    # ── Stage 2 Timing ────────────────────────────────────────────────────
    rtm_lead_blocks:             int   = 3
    captive_buffer_blocks:       int   = 12
    captive_buffer_tolerance_mw: float = 0.5

    # ── Solar SoC Band ────────────────────────────────────────────────────
    soc_solar_low_pct:           float = 0.15   # 15% × 80 = 12.0 MWh
    soc_solar_high_pct:          float = 0.85   # 85% × 80 = 68.0 MWh
    solar_threshold_mw:          float = 0.5
    solar_buffer_blocks:         int   = 2

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def soc_solar_low(self) -> float:
        """Lower SoC bound during solar-generation hours (MWh). = 12.0 MWh"""
        return self.soc_solar_low_pct * self.e_max_mwh

    @property
    def soc_solar_high(self) -> float:
        """Upper SoC bound during solar-generation hours (MWh). = 68.0 MWh"""
        return self.soc_solar_high_pct * self.e_max_mwh

    @property
    def soc_other_low(self) -> float:
        """Lower SoC bound outside solar hours (MWh). = e_min = 8.0 MWh"""
        return self.e_min_mwh

    @property
    def soc_other_high(self) -> float:
        """Upper SoC bound outside solar hours (MWh). = e_max = 80.0 MWh"""
        return self.e_max_mwh

    @property
    def avail_cap_mwh(self) -> float:
        """CERC DSM denominator = S_inv × DT = 16.4 × 0.25 = 4.1 MWh"""
        return self.solar_inverter_mw * 0.25

    @property
    def usable_energy_mwh(self) -> float:
        """Full BESS usable swing = e_max − e_min = 72.0 MWh"""
        return self.e_max_mwh - self.e_min_mwh

    @property
    def rtc_pshort_threshold_mw(self) -> float:
        """
        Fixed penalty threshold = 0.80 × rtc_mw = 4.0 MW.
        This is FIXED against the CONTRACT CEILING (rtc_mw = 5.0 MW),
        NOT against the day's committed level (RTC_committed).
        Used in C_PSHORT and actuals settlement.
        """
        return self.rtc_floor_pct * self.rtc_mw   # = 0.80 × 5.0 = 4.0 MW

    def rtc_band(self, rtc_committed: float) -> Tuple[float, float]:
        """
        ±5% free RT fluctuation band around the Stage-1 committed level.
        captive_rt can move within this band without consumer notification.

        Parameters
        ----------
        rtc_committed : float — MW level chosen by Stage 1 LP.

        Returns
        -------
        (lo, hi) — MW bounds for free RT fluctuation.

        Examples
        --------
        rtc_band(5.0) → (4.75, 5.25)
        rtc_band(3.0) → (2.85, 3.15)
        """
        return (rtc_committed * (1.0 - self.rtc_tol_pct),
                rtc_committed * (1.0 + self.rtc_tol_pct))

    @classmethod
    def from_yaml(cls, path: str) -> "BESSParamsRTC":
        """Load parameters from config/bess_rtc.yaml."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
